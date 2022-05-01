import numpy as np
import matplotlib.pyplot as plt

data = {
    "v_res": 480,
    "h_res": 640,
    "square_side": 0.1,
    "dist": 100.0,
    "eye": [
        300.0,
        500.0,
        0.0
    ],
    "look_at": [
        75.0,
        100.0,
        0.0
    ],
    "up": [
        0.0,
        0.0,
        1.0
    ],
    "background_color": [
        0,
        0,
        0
    ],
    "objects": [
        {
            "color": [
                0,
                0,
                255
            ],
            "ka": 0.1,
            "kd": 0.7,
            "ks": 0.5,
            "exp": 10.0,
            "sphere": {
                "center": [
                    0.0,
                    0.0,
                    0.0
                ],
                "radius": 100.0
            }
        },
        {
            "color": [
                128,
                128,
                128
            ],
            "ka": 0.1,
            "kd": 0.6,
            "ks": 0.1,
            "exp": 40.0,
            "sphere": {
                "center": [
                    150.0,
                    150.0,
                    0.0
                ],
                "radius": 10.0
            }
        }
    ],
    "ambient_light": [
        255,
        255,
        255
    ],
    "lights": [
        {
            "intensity": [
                255,
                255,
                255
            ],
            "position": [
                200.0,
                180.0,
                10.0
            ]
        }
    ]
}


def normalize(vector):
    return vector / np.linalg.norm(vector)


def render(n_linhas, n_colunas, t_l_pixel, d_focal, foco, mira, vetor_up):
    imagem = np.zeros((n_linhas, n_colunas, 3))
    w = normalize(np.array(foco) - np.array(mira))
    u = normalize(np.cross(np.array(vetor_up), w))
    v = np.cross(w, u)
    for i in range(n_linhas):
        for j in range(n_colunas):
            x = t_l_pixel * (j - n_colunas / 2)
            y = -t_l_pixel * (i - n_linhas / 2)
            v_diretor_r = np.multiply(u, x) + np.multiply(v, y) - np.multiply(w, d_focal)
            imagem[i, j] = np.clip(cast(foco, normalize(v_diretor_r)), 0, 1)
    return imagem


def cast(ray_origin, ray_direction):
    c = np.array(data['background_color']) / 255
    t, obj = trace(ray_origin, ray_direction)
    if obj is not None:
        p = np.array(ray_origin) + np.multiply(np.array(ray_direction), t)
        if 'sphere' in obj:
            c = shade(obj, p, -ray_direction, normal_esphere(obj, ray_direction))
        elif 'plane' in obj:
            c = shade(obj, p, -ray_direction, normal_plane(obj, ray_direction))
    return c


def shade(obj, p, w0, n):
    cp = np.multiply(obj['ka'], (np.array(obj['color']) / 255)) * np.array(data['ambient_light'])
    for l in data['lights']:
        lj = normalize(np.array(l['position']) - np.array(p))
        rj = reflect(lj, n)
        pl = np.array(p) + np.dot(10e-5, lj)
        d, s = trace(pl, lj)
        if s is None or np.dot(np.array(lj), (np.array(l['position']) - np.array(p))) < d:
            if np.dot(n, lj) > 0:
                cp = cp + np.multiply(obj['kd'], (np.array(obj['color']) / 255)) * np.multiply(np.dot(n, lj), (np.array(l['intensity'])/255))

            if np.dot(w0, rj) > 0:
                cp = cp + np.multiply(np.multiply(obj['ks'], np.dot(w0, rj) ** obj['exp']), (np.array(l['intensity'])/255))
    return cp


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

    return d, s


def sphere_intersect(center, radius, ray_origin, ray_direction):
    oc = np.array(center) - np.array(ray_origin)
    tca = np.dot(oc, np.array(ray_direction))
    d2 = np.dot(oc, oc) - tca ** 2
    r2 = radius ** 2
    if d2 <= r2:
        thc = np.sqrt((radius ** 2) - d2)
        t0 = tca - thc
        t1 = tca + thc
        if t0 > t1:
            t0, t1 = t1, t0
        if t0 < 0:
            if t1 >= 0:
                return t1
        else:
            return t0
    return None


def plane_intersect(p, n, ray_origin, ray_direction):
    den = np.dot(ray_direction, np.array(n))
    if abs(den) > 1e-6:
        t = np.dot(np.array(p) - ray_origin, n) / den
        if t < 0:
            return np.inf
        else:
            return t
    else:
        return np.inf


def reflect(v, n):
    return 2 * np.dot(v, n) * n - v


def refract(p, w0, n):
    cosi = np.dot(w0, n)
    if cosi < 0:
        n = -n
        cosi = -cosi
    delta = 1 - np.multipy((1 / np.array(n) ** 2), (1 - cosi ** 2))
    if delta < 0:
        return None
    return np.multiply(-1 / np.array(n), w0) - (np.multiply((np.sqrt(delta) - np.multiply((1 / np.array(n)), cosi), np.array(n))))


def normal_esphere(obj, p):
    return normalize(np.array(p) - np.array(obj['sphere']['center']))


def normal_plane(obj, p):
    return normalize(np.array(obj['plane']['normal']))


img = render(data['v_res'], data['h_res'], data['square_side'], data['dist'], data['eye'], data['look_at'], data['up'])
plt.imsave('img.png', img)
plt.imshow(img)
plt.show()
