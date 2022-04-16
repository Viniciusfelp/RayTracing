import numpy as np
import matplotlib.pyplot as plt

data = {'v_res': 480, 'h_res': 640, 'square_side': 0.1, 'dist': 100.0, 'eye': [-100.0, -100.0, 0.0], 'look_at': [0.0, 0.0, 0.0], 'up': [0.0, 0.0, 1.0], 'background_color': [25, 140, 255], 'objects': [{'color': [0, 0, 128], 'plane': {'sample': [0.0, 0.0, -40.0], 'normal': [0.0, 0.0, 1.0]}}, {'color': [243, 229, 243], 'sphere': {'center': [400.0, 200.0, -130.0], 'radius': 100.0}}, {'color': [245, 245, 245], 'sphere': {'center': [30.0, 100.0, -140.0], 'radius': 115.0}}, {'color': [245, 245, 245], 'sphere': {'center': [50.0, 50.0, -110.0], 'radius': 77.0}}, {'color': [230, 250, 230], 'sphere': {'center': [400.0, 500.0, -250.0], 'radius': 240.0}}]}


def normalize(vector):
    return vector / np.linalg.norm(vector)


def render(n_linhas, n_colunas, t_l_pixel, d_focal, foco, mira, vetor_up):
    imagem = np.zeros((n_linhas, n_colunas, 3))
    aux = np.array(mira) - np.array(foco)
    w = normalize(aux)
    u = normalize(np.cross(vetor_up, w))
    v = np.cross(w, u)
    for i in range(n_linhas):
        for j in range(n_colunas):
            x = t_l_pixel * (j - n_colunas / 2)
            y = -t_l_pixel * (i - n_linhas / 2)
            centro = u*x + v*y - w*d_focal
            imagem[i, j] = np.clip(cast(foco, normalize(centro - foco)), 0, 1)
    return imagem


def cast(ray_origin, ray_direction):
    c = data['background_color']
    s, d = trace(ray_origin, ray_direction)
    if s is not None:
        if 'color' in s:
            c = s['color']
    return c


def trace(ray_origin, ray_direction):
    s = None
    d = np.inf
    for obj in data['objects']:
        if 'sphere' in obj:
            t = sphere_intersect(obj['sphere']['center'], obj['sphere']['radius'], ray_origin, ray_direction)
            if t is not None and t < d:
                d = t
                s = obj
        if 'plane' in obj:
            t = plane_intersect(obj['plane']['sample'], obj['plane']['normal'], ray_origin, ray_direction)
            if t is not None and t < d:
                d = t
                s = obj
    return s, d


def sphere_intersect(center, radius, ray_origin, ray_direction):
    oc = np.array(center) - np.array(ray_origin)
    tca = np.dot(oc, ray_direction)
    d2 = np.dot(oc, oc) - tca ** 2
    r2 = radius ** 2
    if d2 <= r2:
        thc = np.sqrt((radius ** 2) - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 < 0:
            t0 = t1
        return t0
    return None


def plane_intersect(p, n, ray_origin, ray_direction):
    den = np.dot(ray_direction, n)
    if abs(den) > 1e-6:
        t = np.dot(np.array(p) - np.array(ray_origin), np.array(n)) / den
        if t >= 0:
            return t
    return None


def main():
    img = render(data['v_res'], data['h_res'], data['square_side'], data['dist'], data['eye'], data['look_at'], data['up'])
    plt.imsave('teste.png', img)
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    main()

