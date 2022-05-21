import math
import numpy as np
import matplotlib.pyplot as plt
import json

with open("espelho2.json", encoding='utf-8') as meu_json:
    data = json.load(meu_json)


def normalize(vector):
    return vector / np.linalg.norm(vector)


def render(n_linhas, n_colunas, t_l_pixel, d_focal, foco, mira, vetor_up, lvl_max):  # funcao para renderizar a imagem
    imagem = np.zeros((n_linhas, n_colunas, 3))  # cria uma matriz de zeros com dimensoes n_linhas x n_colunas x 3
    w = normalize(np.array(foco) - np.array(mira))  # vetor direcao do raio
    u = normalize(np.cross(np.array(vetor_up), w))  # vetor direcao do eixo horizontal
    v = np.cross(w, u)  # vetor direcao do eixo vertical
    for i in range(n_linhas):
        for j in range(n_colunas):
            x = t_l_pixel * (j - n_colunas / 2)  # calcula o valor de x para cada pixel
            y = -t_l_pixel * (i - n_linhas / 2)  # calcula o valor de y para cada pixel
            v_diretor_r = np.multiply(u, x) + np.multiply(v, y) - np.multiply(w, d_focal)
            valor = cast(foco, normalize(v_diretor_r), lvl_max)  # calcula a cor do pixel
            imagem[i, j] = valor / max(*valor, 1)
    return imagem


def cast(ray_origin, ray_direction, k):  # funcao para calcular o valor de cada pixel da imagem
    c = np.array(data['background_color']) / 255
    s, d = trace(ray_origin,ray_direction)  # s = objeto naquele ponto, d = distancia do ponto de intersecao com o objeto
    if s is not None:  # se o raio intercepta algum objeto
        p = ray_origin + d * ray_direction  # p = ponto de intersecao do raio com o objeto
        if 'sphere' in s:
            n = normalize(p - np.array(s['sphere']['center']))
        elif 'plane' in s:
            n = normalize(np.array(s['plane']['normal']))
        obj = (np.array(s['color']) / 255, s['ka'], s['kd'], s['ks'], s['exp'])
        w0 = -1 * ray_direction
        c = shade(obj, p, w0, n)  # c = cor do pixel
        if k > 0:  # se o raio ainda nao chegou ao ponto de origem
            refl = reflect(w0, n)  # vetor direcao do raio refletido
            p_refl = p + refl * (10 ** -5)  # ponto de intersecao do raio refletido
            try:  # se o raio refletido esta dentro do objeto
                if s['kt'] > 0:  # se o objeto tem transparencia
                    eta = s['index_of_refraction']
                    r = refract(eta, w0, n)  # vetor direcao do raio refratado
                    p_linha = p + 10 ** -5 * r  # ponto de intersecao do raio refratado
                    c = c + s['kt'] * cast(p_linha, r, k - 1)  # soma a cor do pixel com a cor do raio refratado
                if s['kr'] > 0:  # se o objeto tem reflexao
                    c = c + s['kr'] * cast(p_refl, refl, k - 1)  # soma a cor do pixel com a cor do raio refletido
            except:  # se o raio refletido esta fora do objeto
                c = c + cast(p_refl, refl, k - 1)

    return c


def shade(obj, p, w0, n):  # funcao para calcular a cor de cada pixel da imagem
    cd, ka, kd, ks, eta = obj  # pega os valores do objeto, cd = cor do objeto, ka = coeficiente ambiente, kd = coeficiente difuso, ks = coeficiente especular, eta = indice de refracao
    cp = np.zeros(3)
    cp += ka * cd * np.array(data['ambient_light']) / 255  # cor ambiente
    for light in data['lights']:
        L = np.array(light['position'])
        lj = normalize(L - p)  # vetor direcao da luz
        cj = np.array(light['intensity']) / 255
        rj = 2 * n * np.inner(lj, n) - lj  # vetor direcao do raio refletido
        ps = p + (10 ** -5) * lj  # ponto de intersecao do raio com a superficie
        s, d = trace(ps, lj)  # s = objeto naquele ponto, d = distancia do ponto de intersecao com o objeto
        if s is None or np.dot(lj, (L - ps)) < d:  # se o ponto de intersecao esta fora do objeto
            if np.dot(n, lj) > 0:  # se o raio esta indo para dentro do objeto
                cp += kd * cd * np.dot(n, lj) * cj  # soma a cor do pixel com a cor da luz
            if np.dot(w0, rj) > 0:  # se o raio refletido esta indo para dentro do objeto
                cp += ks * (np.dot(w0, rj) ** eta) * cj  # soma a cor do pixel com a cor do raio refletido

    return cp


def trace(ray_origin, ray_direction):  # funcao para calcular o objeto que o raio intercepta
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


def sphere_intersect(center, radius, ray_origin,ray_direction):  # funcao para calcular a interseccao entre o raio e a esfera
    oc = np.array(center) - np.array(ray_origin)
    tca = np.dot(oc, np.array(ray_direction))
    d2 = np.dot(oc, oc) - tca ** 2
    r2 = radius ** 2
    if d2 <= r2:  # se o raio intercepta a esfera
        thc = np.sqrt((radius ** 2) - d2)  # calcula o raio que intercepta a esfera
        t0 = tca - thc
        t1 = tca + thc
        if t0 > t1:  # se o raio intercepta a esfera em dois pontos
            t0, t1 = t1, t0
        if t0 < 0:  # se o raio intercepta a esfera antes do raio
            if t1 >= 0:  # se o raio intercepta a esfera depois do raio
                return t1
        else:  # se o raio intercepta a esfera depois do raio
            return t0
    return None


def plane_intersect(p, n, ray_origin, ray_direction):  # funcao para calcular a interseccao entre o raio e o plano
    den = np.dot(ray_direction, np.array(n))
    if abs(den) > 1e-6:  # se o raio nao esta paralelo ao plano
        t = np.dot(np.array(p) - ray_origin, n) / den  # calcula o raio que intercepta o plano
        if t <= 0:  # se o raio intercepta o plano antes do raio
            return np.inf
        else:  # se o raio intercepta o plano depois do raio
            return t
    else:  # se o raio esta paralelo ao plano
        return np.inf


def refract(eta, w0, n):  # funcao para calcular o raio refletido
    cos_i = np.dot(w0, n)  # calcula o cos do angulo entre o raio e a normal
    if cos_i < 0:  # se o raio esta indo para fora do objeto - o tringulo é obtuso
        n = -n
        eta = 1 / eta
        cos_i = -cos_i
    delta = 1 - (1 / (eta ** 2)) * (1 - cos_i ** 2)
    if delta < 0:  # se o raio refletido esta indo para fora do objeto
        raise Exception("Erro")
    else:
        return (-1 / eta) * w0 - (math.sqrt(delta) - (1 / eta) * cos_i) * n  # vetor direcao do raio refletido


def reflect(l, n):  # funcao para calcular o raio refletido
    return 2 * np.dot(n, l) * n - l


img = render(data['v_res'], data['h_res'], data['square_side'], data['dist'], data['eye'], data['look_at'], data['up'],
             data['max_depth'])
# plt.imsave('img.png', img)
plt.imshow(img)
plt.show()
