#!/usr/bin/env python3
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    Copyright (C) 2008  Nick Redshaw
#    Copyright (C) 2018  Francisco Sanchez Arroyo
#

# TODO
# safe area on new life
# sounds thump

# Notes:
# random.randrange returns an int
# random.uniform returns a float
# p for pause
# j for toggle showing FPS
# o for frame advance whilst paused

import pygame
import sys
import os
import random
from pygame.locals import *
from .util.vectorsprites import *
from .ship import *
from .stage import *
from .badies import *
from .shooter import *
from .soundManager import *

class Asteroids():

    explodingTtl = 180
    
    def __init__(self, headless=False):
        self.headless = headless
        
        self.debug_mode = True  # debug visual
        self.last_action = "none"
        self.last_reward = 0.0
        self.last_angle_diff = 0.0
        
        pygame.init()
        pygame.font.init() 

        if not headless:
            # normal windows
            self.stage = Stage('Atari Asteroids', (1200, 630))
        else:
            # without windows (simulator)
            pygame.display.init()
            self.stage = type('StageMock', (), {})()
            self.stage.width = 1200
            self.stage.height = 630
            self.stage.screen = pygame.Surface((1, 1))
            self.stage.spriteList = []
            self.stage.addSprite = lambda *args, **kwargs: None
            self.stage.removeSprite = lambda *args, **kwargs: None
            self.stage.moveSprites = lambda *args, **kwargs: None
            self.stage.drawSprites = lambda *args, **kwargs: None

        # --- other atributes ---
        self.paused = False
        self.showingFPS = False
        self.frameAdvance = False
        self.gameState = "playing"
        self.rockList = []
        self.createRocks(1)
        self.saucer = None
        self.secondsCount = 1
        self.score = 0
        self.ship = None
        self.lives = 0


    def initialiseGame(self):
        self.gameState = 'playing'

        # Remove old rocks
        for sprite in list(self.rockList):
            if sprite in self.stage.spriteList:
                self.stage.removeSprite(sprite)

        # Remove old saucer
        if self.saucer is not None:
            self.killSaucer()

        # Remove any existing ship & its jet (defensive clean-up)
        if self.ship is not None:
            if getattr(self.ship, "thrustJet", None) in self.stage.spriteList:
                self.stage.removeSprite(self.ship.thrustJet)
            if self.ship in self.stage.spriteList:
                self.stage.removeSprite(self.ship)

        # Optionally remove any stray debris/bullets that may linger
        # (depends on your Stage implementation; if you tag sprites by type, filter them here)

        self.startLives = 1
        self.score = 0
        self.rockList = []
        self.numRocks = 3
        self.nextLife = 10000
        self.secondsCount = 1

        self.createNewShip()
        self.createLivesList()
        self.createRocks(self.numRocks)


    def createNewShip(self):
        # If there was a previous ship, remove it and its thrust jet
        if self.ship is not None:
            # remove debris of the old ship if any
            for debris in getattr(self.ship, "shipDebrisList", []):
                if debris in self.stage.spriteList:
                    self.stage.spriteList.remove(debris)
            # remove old ship and jet if still present
            if self.ship in self.stage.spriteList:
                self.stage.removeSprite(self.ship)
            if getattr(self.ship, "thrustJet", None) in self.stage.spriteList:
                self.stage.removeSprite(self.ship.thrustJet)

        # Create the new ship
        self.ship = Ship(self.stage)
        self.stage.addSprite(self.ship.thrustJet)
        self.stage.addSprite(self.ship)


    def createLivesList(self):
        self.lives += 1
        self.livesList = []
        for i in range(1, self.startLives):
            self.addLife(i)

    def addLife(self, lifeNumber):
        self.lives += 1
        ship = Ship(self.stage)
        self.stage.addSprite(ship)
        ship.position.x = self.stage.width - \
            (lifeNumber * ship.boundingRect.width) - 10
        ship.position.y = 0 + ship.boundingRect.height
        self.livesList.append(ship)

    def createRocks(self, numRocks):
        for _ in range(0, numRocks):
            position = Vector2d(random.randrange(-10, 10),
                                random.randrange(-10, 10))

            newRock = Rock(self.stage, position, Rock.largeRockType)
            self.stage.addSprite(newRock)
            self.rockList.append(newRock)

    def update_one_frame(self):
        self.secondsCount += 1
        self.input(pygame.event.get())
        self.stage.screen.fill((10, 10, 10))
        self.stage.moveSprites()
        self.stage.drawSprites()
        
        self.debug_draw()
        
        self.draw_line_to_enemy_ship()
        
        self.doSaucerLogic()
        self.displayScore()
        if self.showingFPS:
            self.displayFps()
        self.checkScore()
    

        if self.gameState == 'playing':
            self.playing()
        elif self.gameState == 'exploding':
            self.initialiseGame()
        else:
            self.displayText()
            
        pygame.display.flip()

    def playing(self):
        if self.lives == 0:
            self.gameState = 'attract_mode'
        else:
            self.processKeys()
            self.checkCollisions()
            if len(self.rockList) == 0:
                self.levelUp()

    def doSaucerLogic(self):
        if self.saucer is not None:
            if self.saucer.laps >= 2:
                self.killSaucer()

        # Create a saucer
        if self.secondsCount % 2000 == 0 and self.saucer is None:
            randVal = random.randrange(0, 10)
            if randVal <= 3:
                self.saucer = Saucer(
                    self.stage, Saucer.smallSaucerType, self.ship)
            else:
                self.saucer = Saucer(
                    self.stage, Saucer.largeSaucerType, self.ship)
            self.stage.addSprite(self.saucer)

    def exploding(self):
        self.explodingCount += 1
        if self.explodingCount > self.explodingTtl:
            self.gameState = 'playing'
            [self.stage.spriteList.remove(debris)
             for debris in self.ship.shipDebrisList]
            self.ship.shipDebrisList = []
            self.ship.visible = False
            self.createNewShip()

    def levelUp(self):
        self.numRocks += 1
        self.createRocks(self.numRocks)

    # move this kack somewhere else!
    def displayText(self):
        font1 = pygame.font.Font('../res/Hyperspace.otf', 50)
        font2 = pygame.font.Font('../res/Hyperspace.otf', 20)
        font3 = pygame.font.Font('../res/Hyperspace.otf', 30)

        titleText = font1.render('Asteroids', True, (180, 180, 180))
        titleTextRect = titleText.get_rect(centerx=self.stage.width/2)
        titleTextRect.y = self.stage.height/2 - titleTextRect.height*2
        self.stage.screen.blit(titleText, titleTextRect)

        keysText = font2.render(
            '(C) 1979 Atari INC.', True, (255, 255, 255))
        keysTextRect = keysText.get_rect(centerx=self.stage.width/2)
        keysTextRect.y = self.stage.height - keysTextRect.height - 20
        self.stage.screen.blit(keysText, keysTextRect)

        instructionText = font3.render(
            'Press start to Play', True, (200, 200, 200))
        instructionTextRect = instructionText.get_rect(
            centerx=self.stage.width/2)
        instructionTextRect.y = self.stage.height/2 - instructionTextRect.height
        self.stage.screen.blit(instructionText, instructionTextRect)

    def displayScore(self):
        font1 = pygame.font.Font('../res/Hyperspace.otf', 30)
        scoreStr = str("%02d" % self.score)
        scoreText = font1.render(scoreStr, True, (200, 200, 200))
        scoreTextRect = scoreText.get_rect(centerx=100, centery=45)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def displayPaused(self):
        if self.paused:
            font1 = pygame.font.Font('../res/Hyperspace.otf', 30)
            pausedText = font1.render("Paused", True, (255, 255, 255))
            textRect = pausedText.get_rect(
                centerx=self.stage.width/2, centery=self.stage.height/2)
            self.stage.screen.blit(pausedText, textRect)
            pygame.display.update()

    # Should move the ship controls into the ship class
    def input(self, events):
        self.frameAdvance = False
        for event in events:
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    sys.exit(0)
                if self.gameState == 'playing':
                    if event.key == K_SPACE:
                        self.ship.fireBullet()
                    elif event.key == K_b:
                        self.ship.fireBullet()
                    elif event.key == K_h:
                        self.ship.enterHyperSpace()
                elif self.gameState == 'attract_mode': 
                    # Start a new game
                    if event.key == K_RETURN:
                        self.initialiseGame()

                if event.key == K_p:
                    if self.paused:  # (is True)
                        self.paused = False
                    else:
                        self.paused = True

                if event.key == K_j:
                    if self.showingFPS:  # (is True)
                        self.showingFPS = False
                    else:
                        self.showingFPS = True

                if event.key == K_f:
                    pygame.display.toggle_fullscreen()

                if event.key == K_k:
                    self.killShip()
                    
            elif event.type == KEYUP:
                if event.key == K_o:
                    self.frameAdvance = True

    def processKeys(self):
        key = pygame.key.get_pressed()

        if key[K_LEFT] or key[K_z]:
            self.ship.rotateLeft()
        elif key[K_RIGHT] or key[K_x]:
            self.ship.rotateRight()

        if key[K_UP] or key[K_n]:
            self.ship.increaseThrust()
            self.ship.thrustJet.accelerating = True
        else:
            self.ship.thrustJet.accelerating = False


    # Check for ship hitting the rocks etc.
    def checkCollisions(self):

        newRocks = []
        shipHit, saucerHit = False, False

        # Rocks
        for rock in self.rockList:
            rockHit = False

            if not self.ship.inHyperSpace and rock.collidesWith(self.ship):
                p = rock.checkPolygonCollision(self.ship)
                if p is not None:
                    shipHit = True
                    rockHit = True

            if self.saucer is not None:
                if rock.collidesWith(self.saucer):
                    saucerHit = True
                    rockHit = True

                if self.saucer.bulletCollision(rock):
                    rockHit = True

                if self.ship.bulletCollision(self.saucer):
                    saucerHit = True
                    self.score += self.saucer.scoreValue

            if self.ship.bulletCollision(rock):
                rockHit = True

            if rockHit:
                self.rockList.remove(rock)
                self.stage.spriteList.remove(rock)

                if rock.rockType == Rock.largeRockType:
                    # playSound("explode1")
                    newRockType = Rock.mediumRockType
                    self.score += 50
                elif rock.rockType == Rock.mediumRockType:
                    # playSound("explode2")
                    newRockType = Rock.smallRockType
                    self.score += 100
                else:
                    # playSound("explode3")
                    self.score += 200

                if rock.rockType != Rock.smallRockType:
                    # new rocks
                    for _ in range(0, 2):
                        position = Vector2d(rock.position.x, rock.position.y)
                        newRock = Rock(self.stage, position, newRockType)
                        self.stage.addSprite(newRock)
                        self.rockList.append(newRock)

                self.createDebris(rock)

        # Saucer bullets
        if self.saucer is not None:
            if not self.ship.inHyperSpace:
                if self.saucer.bulletCollision(self.ship):
                    shipHit = True
                    self.ship.bulletShot += 1

                if self.saucer.collidesWith(self.ship):
                    shipHit = True
                    saucerHit = True

            if saucerHit:
                self.createDebris(self.saucer)
                self.killSaucer()
                self.ship.bulletHit += 1

        if shipHit:
            self.killShip()

    def killShip(self):
        # stopSound("thrust")
        # playSound("explode2")
        self.explodingCount = 0
        self.lives -= 1
        if self.livesList:
            ship_icon = self.livesList.pop()
            self.stage.removeSprite(ship_icon)

        # Remove current ship and jet
        if getattr(self.ship, "thrustJet", None) in self.stage.spriteList:
            self.stage.removeSprite(self.ship.thrustJet)
        if self.ship in self.stage.spriteList:
            self.stage.removeSprite(self.ship)

        self.gameState = 'exploding'
        self.ship.explode()


    def killSaucer(self):
        # stopSound("lsaucer")
        # stopSound("ssaucer")
        # playSound("explode2")
        self.stage.removeSprite(self.saucer)
        self.saucer = None

    def createDebris(self, sprite):
        for _ in range(0, 25):
            position = Vector2d(sprite.position.x, sprite.position.y)
            debris = Debris(position, self.stage)
            self.stage.addSprite(debris)

    def displayFps(self):
        font2 = pygame.font.Font('../res/Hyperspace.otf', 15)
        fpsStr = str(self.fps)+(' FPS')
        scoreText = font2.render(fpsStr, True, (255, 255, 255))
        scoreTextRect = scoreText.get_rect(
            centerx=(self.stage.width/2), centery=15)
        self.stage.screen.blit(scoreText, scoreTextRect)

    def checkScore(self):
        if self.score > 0 and self.score > self.nextLife:
            # playSound("extralife")
            self.nextLife += 10000
            self.addLife(self.lives)
            
    def draw_line_to_enemy_ship(self):

        if not self.ship or not self.saucer:
            return

        ship_pos = (int(self.ship.position.x), int(self.ship.position.y))
        enemy_pos = (int(self.saucer.position.x), int(self.saucer.position.y))
        pygame.draw.line(self.stage.screen, (255, 0, 0), ship_pos, enemy_pos, 2)

    def debug_draw(self):
        if not self.debug_mode:
            return

        ship = self.ship
        surface = self.stage.screen
        cx, cy = int(ship.position.x), int(ship.position.y)

        if self.rockList:
            nearest = min(self.rockList,
                        key=lambda r: (r.position.x - ship.position.x)**2 +
                                        (r.position.y - ship.position.y)**2
                        )
            
            nx, ny = int(nearest.position.x), int(nearest.position.y)

            angle_color = self.compute_angle_color(self.last_angle_diff)
            
            pygame.draw.line(surface, angle_color, (cx, cy), (nx, ny), 3)

        angle_rad = -math.radians(ship.angle)
        length = 100
        ax = cx + length * math.sin(angle_rad)
        ay = cy - length * math.cos(angle_rad)
        pygame.draw.line(surface, (0, 180, 255), (cx, cy), (ax, ay), 2)

        fov = math.radians(30)
        left_angle = angle_rad - fov
        right_angle = angle_rad + fov

        lx = cx + 120 * math.sin(left_angle)
        ly = cy - 120 * math.cos(left_angle)
        rx = cx + 120 * math.sin(right_angle)
        ry = cy - 120 * math.cos(right_angle)

        pygame.draw.line(surface, (100, 100, 255), (cx, cy), (lx, ly), 1)
        pygame.draw.line(surface, (100, 100, 255), (cx, cy), (rx, ry), 1)

        # --- HUD con transparencia ---
        hud = pygame.Surface((300, 100), pygame.SRCALPHA)
        hud.fill((0, 0, 0, 130))

        font = pygame.font.Font(None, 26)
        text1 = font.render(f"Acción IA: {self.last_action}", True, (255,255,255))
        text2 = font.render(f"Reward: {self.last_reward:.2f}", True, (255,255,255))
        text3 = font.render(f"Alineación: {self.last_angle_diff:.2f}", True, (255,255,150))

        hud.blit(text1, (10, 10))
        hud.blit(text2, (10, 40))
        hud.blit(text3, (10, 70))

        hud_x = surface.get_width() - hud.get_width() - 10
        hud_y = 10

        surface.blit(hud, (hud_x, hud_y))

    def compute_angle_color(self, angle_diff):
        t = min(angle_diff / math.pi, 1.0)
        r = int(255 * t)
        g = int(255 * (1 - t))
        return (r, g, 0)

# Script to run the game
if not pygame.font:
    print('Warning, fonts disabled')
if not pygame.mixer:
    print('Warning, sound disabled')

initSoundManager()
