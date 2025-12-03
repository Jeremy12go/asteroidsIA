import sys, os, math
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.asteroids import Asteroids
import numpy as np

class Environment:
    
    def __init__(self):
        self.game = Asteroids(False)
        self.game.initialiseGame()
        self.done = False

        # Previous ship state to measure movement patterns
        ship = self.game.ship
        self.prev_angle = ship.angle
        self.prev_vx = ship.heading.x
        self.prev_vy = ship.heading.y
        self.straight_frames = 0

        self.last_action = None
        self.last_reward = 0.0

    def reset(self):
        self.game.initialiseGame()
        self.done = False
        return self.get_state()

    def step(self, action): # net parameter
        
        self.game.current_net = None  # no needed for DQN visualization
        self.game.last_action = action

        # 1. Aplicar acción de la IA (giro, thrust, disparo)
        self.apply_action(action)

        # 2. Avanzar un frame del juego
        self.game.update_one_frame()

        # 3. Calcular el reward del frame actual
        reward = self.compute_reward()

        # 4. Guardar acción y reward en el HUD debug
        # self.game.last_action = action
        self.game.last_reward = reward

        # 5. Calcular la diferencia angular para visualización
        ship = self.game.ship
        nearest = self.get_nearest_asteroid(ship.position.x, ship.position.y)

        if nearest:
            dx = nearest.position.x - ship.position.x
            dy = nearest.position.y - ship.position.y

            # a qué ángulo está el asteroide (ángulo matemático estándar)
            angle_to_asteroid = math.atan2(dy, dx)

            # calcular vector de dirección de la nave
            angle_rad = math.radians(ship.angle)
            dir_x = -math.sin(angle_rad)
            dir_y = -math.cos(angle_rad)

            # convertir el vector a un ángulo compatible con atan2
            ship_angle = math.atan2(dir_y, dir_x)

            # diferencia angular mínima
            raw_diff = abs(angle_to_asteroid - ship_angle)
            angle_diff = min(raw_diff, 2*math.pi - raw_diff)
        else:
            angle_diff = 0.0

        # Guardar este valor para dibujarlo en la ventana
        self.game.last_angle_diff = angle_diff

        # 6. Revisar si el episodio terminó
        done = (self.game.gameState == 'exploding')

        # 7. Devolver el estado nuevo
        state = self.get_state()
        self.game.current_net = None # net value
        self.game.current_state = state
        return state, reward, done


    def get_state(self):
        ship = self.game.ship
        
        x = ship.position.x / self.game.stage.width
        y = ship.position.y / self.game.stage.height

        vx = ship.heading.x / ship.maxVelocity
        vy = ship.heading.y / ship.maxVelocity

        angle_norm = ship.angle / 360.0

        # --- Nearest asteroid ---
        nearest = self.get_nearest_asteroid(ship.position.x, ship.position.y)
        if nearest:
            dx = nearest.position.x - ship.position.x
            dy = nearest.position.y - ship.position.y

            dx_norm = dx / self.game.stage.width
            dy_norm = dy / self.game.stage.height

            distance = np.sqrt(dx_norm**2 + dy_norm**2)

            # Dirección hacia el asteroide (ángulo matemático estándar)
            angle_to_asteroid = math.atan2(dy, dx)

            # Dirección real de la nave
            angle_rad = math.radians(ship.angle)
            ship_dir_x = -math.sin(angle_rad)
            ship_dir_y = -math.cos(angle_rad)
            ship_angle = math.atan2(ship_dir_y, ship_dir_x)

            # Diferencia angular mínima
            raw_diff = abs(angle_to_asteroid - ship_angle)
            angle_diff = min(raw_diff, 2*math.pi - raw_diff)

            # Normalizar dif. angular a 0–1
            angle_diff_norm = angle_diff / math.pi

        else:
            dx_norm = dy_norm = 0
            distance = 1
            angle_to_asteroid = 0
            angle_diff_norm = 1   # "máximo desalineado"

        # --- Enemy (saucer) data ---
        if self.game.saucer is not None:
            se = self.game.saucer
            x_enemy = se.position.x / self.game.stage.width
            y_enemy = se.position.y / self.game.stage.height
            vx_enemy = se.heading.x / 3.0
            vy_enemy = se.heading.y / 3.0
        else:
            x_enemy = y_enemy = vx_enemy = vy_enemy = 0.0

        return np.array([
            x, y,
            vx, vy,
            angle_norm,

            dx_norm, dy_norm, distance,
            angle_to_asteroid, angle_diff_norm,
            x_enemy, y_enemy,
            vx_enemy, vy_enemy

        ], dtype=float)

    def apply_action(self, action):
        if action == 0: self.game.ship.rotateLeft()
        elif action == 1: self.game.ship.rotateRight()
        elif action == 2: self.game.ship.increaseThrust()
        elif action == 3: self.game.ship.decreaseThrust()
        elif action == 4: self.game.ship.fireBullet()

    def compute_reward(self):
        ship = self.game.ship
        reward = 0.1  # reward por sobrevivir

        # ============================
        # 1) MOVIMIENTO REGULAR / RARO
        # ============================

        angle_change = abs(ship.angle - self.prev_angle)
        dvx = abs(ship.heading.x - self.prev_vx)
        dvy = abs(ship.heading.y - self.prev_vy)

        # --- recto demasiado tiempo ---
        """
        if angle_change < 1.0 and dvx < 0.01 and dvy < 0.01:
            self.straight_frames += 1
        else:
            self.straight_frames = 0

        if self.straight_frames > 60:
            reward -= 0.03  # ligera penalización
        """
        # --- movimiento raro (cambios bruscos) ---
        """
        movement_change = angle_change * 0.005 + (dvx + dvy) * 0.5
        if movement_change > 1.0:
            reward -= movement_change  # penalización proporcional
        """
        # ==========================================
        # 2) CALCULAR ALINEACIÓN ANGULAR REAL 
        # ==========================================

        nearest = self.get_nearest_asteroid(ship.position.x, ship.position.y)
        if nearest:
            dx = nearest.position.x - ship.position.x
            dy = nearest.position.y - ship.position.y

            angle_to_asteroid = math.atan2(dy, dx)
            ship_angle_rad = math.radians(ship.angle)

            # error angular absoluto en [-pi, +pi]
            angle_diff = abs((angle_to_asteroid - ship_angle_rad + math.pi) % (2*math.pi) - math.pi)

            # normalizar 0..1
            angle_diff_norm = min(angle_diff / math.pi, 1.0)

            # -------------------------------------------------
            #  REWARD 2.1: ALINEACIÓN (apuntar bien = recompensa)
            # -------------------------------------------------
            reward += (1 - angle_diff_norm) * 0.5    # máx +0.5

            # -------------------------------------------------
            #  REWARD 2.2: DISPARO BIEN ALINEADO
            # -------------------------------------------------
            if self.last_action == 4 and angle_diff_norm < 0.05:
                reward += 5   # fuerte refuerzo de disparo correcto

            # -------------------------------------------------
            #  PENALTY 2.3: DISPARO INÚTIL (mal alineado)
            # -------------------------------------------------
            if self.last_action == 4 and angle_diff_norm > 0.2:
                reward -= 3

            # ------------------------------------------
            # PENALTY 2.4: No alinearse nunca
            # ------------------------------------------
            if angle_diff_norm > 0.50:
                reward -= 0.05

            # ========================
            # 3) PENALIDAD POR DISTANCIA
            # ========================
            dist = math.sqrt(dx**2 + dy**2)
            if dist < 100:    # demasiado cerca de un asteroide
                reward -= 0.5

        # ==========================
        # 4) ASTEROIDES DESTRUIDOS
        # ==========================
        reward += ship.bulletHit * 20

        # ==========================
        # 5) PENALIZAR SPAM DE DISPAROS
        # ==========================
        reward -= ship.bulletShot * 0.2

        # ==========================
        # 6) MUERTE
        # ==========================
        if self.game.gameState == 'exploding':
            reward -= 40

        # ==========================
        # 7) GUARDAR ESTADO PREVIO
        # ==========================
        self.prev_angle = ship.angle
        self.prev_vx = ship.heading.x
        self.prev_vy = ship.heading.y

        # ==========================
        # 8) RESET DE CONTADORES
        # ==========================
        ship.bulletHit = 0
        ship.bulletShot = 0

        return reward


    def get_nearest_asteroid(self, sx, sy):
        if not self.game.rockList:
            return None
        return min(self.game.rockList,
                   key=lambda a: (a.position.x - sx)**2 + (a.position.y - sy)**2)
