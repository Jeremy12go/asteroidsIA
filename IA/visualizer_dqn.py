import pygame
import numpy as np

def draw_dqn(surface, model, inputs,
             top_left=(830, 20),     # MÁS PEGADO AL LADO DERECHO
             layer_width=90,         # Distancia horizontal reducida
             node_radius=6):         # Nodos más pequeños
    """
    Dibuja una red neuronal simple reducida para ventana 1200x630.
    """

    # -------------------------
    # 1. FORWARD CON DEBUG
    # -------------------------
    with np.errstate(all='ignore'):
        x = np.array(inputs, dtype=np.float32)
        x = np.nan_to_num(x)
        x_t = x.reshape(1, -1)

    # Extraer pesos
    w1 = model.net[0].weight.data.cpu().numpy()
    b1 = model.net[0].bias.data.cpu().numpy()
    w2 = model.net[2].weight.data.cpu().numpy()
    b2 = model.net[2].bias.data.cpu().numpy()

    # Activaciones
    h1 = np.tanh((x_t @ w1.T) + b1)
    out = (h1 @ w2.T) + b2

    h1 = h1.flatten()
    out = out.flatten()

    # -------------------------
    # 2. POSICIONES
    # -------------------------
    x0, y0 = top_left

    n_in = len(x)
    n_h = len(h1)
    n_out = len(out)

    max_nodes = max(n_in, n_h, n_out)
    v_spacing = 14  # Más compacto
    total_height = (max_nodes - 1) * v_spacing

    def layer_positions(n_nodes, layer_index):
        X = x0 + layer_index * layer_width
        start_y = y0 + (total_height / 2) - ((n_nodes - 1) * v_spacing) / 2
        return [(X, start_y + i * v_spacing) for i in range(n_nodes)]

    pos_in = layer_positions(n_in, 0)
    pos_h1 = layer_positions(n_h, 1)
    pos_out = layer_positions(n_out, 2)

    # -------------------------
    # 3. FUNCIONES DE COLOR
    # -------------------------
    def node_color(value):
        try:
            value = float(value)
        except:
            value = 0.0

        if np.isnan(value) or np.isinf(value):
            value = 0.0

        v = int((value + 1) / 2 * 255)
        v = max(0, min(255, v))

        if value >= 0:
            return (255, 255, v)
        else:
            return (v, v, 255)

    def weight_color(w):
        if w >= 0:
            return (0, 200, 0)
        else:
            return (220, 40, 40)

    def weight_width(w):
        return max(1, int(min(4, abs(w) * 2)))

    # -------------------------
    # 4. DIBUJAR CONEXIONES
    # -------------------------

    # input → h1
    for i, (x1, y1) in enumerate(pos_in):
        for j, (x2, y2) in enumerate(pos_h1):
            w = w1[j][i]
            pygame.draw.line(surface, weight_color(w),
                             (x1, y1), (x2, y2),
                             weight_width(w))

    # h1 → out
    for j, (xh, yh) in enumerate(pos_h1):
        for k, (xo, yo) in enumerate(pos_out):
            w = w2[k][j]
            pygame.draw.line(surface, weight_color(w),
                             (xh, yh), (xo, yo),
                             weight_width(w))

    # -------------------------
    # 5. NODOS
    # -------------------------
    # input nodes
    for i, (x, y) in enumerate(pos_in):
        pygame.draw.circle(surface, node_color(x), (int(x), int(y)), node_radius)
        pygame.draw.circle(surface, (0,0,0), (int(x),int(y)), node_radius, 1)

    # hidden nodes
    for j, (x, y) in enumerate(pos_h1):
        pygame.draw.circle(surface, node_color(h1[j]), (int(x), int(y)), node_radius)
        pygame.draw.circle(surface, (0,0,0), (int(x),int(y)), node_radius, 1)

    # output nodes
    for k, (x, y) in enumerate(pos_out):
        pygame.draw.circle(surface, node_color(out[k]), (int(x), int(y)), node_radius)
        pygame.draw.circle(surface, (0,0,0), (int(x),int(y)), node_radius, 1)
