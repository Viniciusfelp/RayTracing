
import numpy as np
import matplotlib.pyplot as plt


def normalize(vector):
    return vector / np.linalg.norm(vector)


def render(n_linhas, n_colunas, t_l_pixel, d_focal, foco, mira, vetor_up):
    w = normalize(mira - foco)
    u = normalize(np.cross(vetor_up, w))
    v = np.cross(w, u)
    for i in range(n_linhas):
        for j in range(n_colunas):
            x = t_l_pixel * (j - n_colunas / 2)
            y = -t_l_pixel * (i - n_linhas / 2)
            centro = np.dot(u, x) + np.dot(v, y) - np.dot(w, d_focal)
            imagem = cast(foco, normalize(centro - foco))
            imagem[i, j] = imagem
    return imagem


def cast(ray_origin, ray_direction):
    c = np.zeros(3)
    obj = trace(ray_origin, ray_direction)
    if obj is not None:
        c = obj
    return c


def trace(objects, ray_origin, ray_direction):
    s = None
    d = np.inf
    for object in objects:
        try:
            t = sphere_intersect(object.center, object.radius, ray_origin, ray_direction)
            if t is not None and t < d:
                s = object
                d = t

        except:

            t = plane_intersect(object.center, object.normal, ray_origin, ray_direction)
            if t is not None and t < d:
                s = object
                d = t
    return s, d


def sphere_intersect(center, radius, ray_origin, ray_direction):
    l = center - ray_origin
    tca = np.dot(l, ray_direction)
    l2 = np.dot(l, l)
    d2 = l2 - tca ** 2
    if d2 > radius ** 2:
        return None
    thc = np.sqrt(radius ** 2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    if t0 < 0:
        t0 = t1
    if t0 < 0:
        return None
    return t0, t1


def plane_intersect(p, n, ray_origin, ray_direction):
    den = np.dot(ray_direction, n)
    if abs(den) < 1e-6:
        t = -np.dot(p - ray_origin, n) / den
        if t < 0:
            return None
        else:
            return t
    else:
        return None



