from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)


def normalize_weights(criteria):
    """Zwraca tablicę wag znormalizowanych do sumy 1."""
    weights = np.array([c['weight'] for c in criteria])
    if np.sum(weights) == 0:
        return np.ones_like(weights) / len(weights)
    return weights / np.sum(weights)


def saw_method(variants, criteria, matrix):
    """Metoda SAW (Simple Additive Weighting)."""
    X = np.array(matrix, dtype=float)
    weights = normalize_weights(criteria)
    types = [c['type'] for c in criteria]

    norm_X = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        if types[j] == 'profit':
            norm_X[:, j] = col / np.max(col) if np.max(col) != 0 else col
        else:
            norm_X[:, j] = np.min(col) / col if np.min(col) != 0 else col

    scores = np.sum(norm_X * weights, axis=1)
    return scores


def topsis_method(variants, criteria, matrix):
    """Metoda TOPSIS."""
    X = np.array(matrix, dtype=float)
    weights = normalize_weights(criteria)
    types = [c['type'] for c in criteria]

    norm_X = X / np.sqrt(np.sum(X ** 2, axis=0))
    weighted = norm_X * weights

    ideal = []
    anti_ideal = []
    for j in range(weighted.shape[1]):
        col = weighted[:, j]
        if types[j] == 'profit':
            ideal.append(np.max(col))
            anti_ideal.append(np.min(col))
        else:
            ideal.append(np.min(col))
            anti_ideal.append(np.max(col))

    ideal = np.array(ideal)
    anti_ideal = np.array(anti_ideal)

    dist_to_ideal = np.sqrt(np.sum((weighted - ideal) ** 2, axis=1))
    dist_to_anti = np.sqrt(np.sum((weighted - anti_ideal) ** 2, axis=1))

    scores = dist_to_anti / (dist_to_ideal + dist_to_anti)
    return scores


def vikor_method(variants, criteria, matrix, v=0.5):
    """
    Metoda VIKOR.
    v - waga strategii (0.5 to kompromis między użytecznością a żalem)
    """
    X = np.array(matrix, dtype=float)
    weights = normalize_weights(criteria)
    types = [c['type'] for c in criteria]

    n_variants, n_criteria = X.shape

    # Wyznaczenie najlepszych i najgorszych wartości
    f_star = np.zeros(n_criteria)
    f_minus = np.zeros(n_criteria)

    for j in range(n_criteria):
        col = X[:, j]
        if types[j] == 'profit':
            f_star[j] = np.max(col)
            f_minus[j] = np.min(col)
        else:
            f_star[j] = np.min(col)
            f_minus[j] = np.max(col)

    # Obliczenie S (użyteczność) i R (żal)
    S = np.zeros(n_variants)
    R = np.zeros(n_variants)

    for i in range(n_variants):
        s_i = 0
        r_i = 0
        for j in range(n_criteria):
            diff = f_star[j] - X[i, j]
            denom = f_star[j] - f_minus[j]
            if denom == 0:
                normalized = 0
            else:
                normalized = diff / denom
            weighted_norm = weights[j] * normalized
            s_i += weighted_norm
            r_i = max(r_i, weighted_norm)
        S[i] = s_i
        R[i] = r_i

    # Obliczenie Q
    S_star = np.min(S)
    S_minus = np.max(S)
    R_star = np.min(R)
    R_minus = np.max(R)

    Q = np.zeros(n_variants)
    for i in range(n_variants):
        if S_minus - S_star == 0:
            s_term = 0
        else:
            s_term = (S[i] - S_star) / (S_minus - S_star)
        if R_minus - R_star == 0:
            r_term = 0
        else:
            r_term = (R[i] - R_star) / (R_minus - R_star)
        Q[i] = v * s_term + (1 - v) * r_term

    # VIKOR zwraca wynik Q (im mniejszy tym lepszy)
    # Dla spójności z innymi metodami (gdzie większy = lepszy) zwracamy -Q lub 1-Q
    # Wybieramy 1-Q, aby większa wartość oznaczała lepszy wariant
    return 1 - Q


def copras_method(variants, criteria, matrix):
    """
    Metoda COPRAS (COmplex PRoportional ASsessment).
    """
    X = np.array(matrix, dtype=float)
    weights = normalize_weights(criteria)
    types = [c['type'] for c in criteria]

    n_variants, n_criteria = X.shape

    # Normalizacja liniowa (suma kolumny = 1)
    norm_X = np.zeros_like(X)
    for j in range(n_criteria):
        col_sum = np.sum(X[:, j])
        if col_sum != 0:
            norm_X[:, j] = X[:, j] / col_sum
        else:
            norm_X[:, j] = X[:, j]

    # Ważona znormalizowana macierz
    weighted = norm_X * weights

    # Sumy dla kryteriów zysk i koszt
    S_plus = np.zeros(n_variants)  # zysk (profit)
    S_minus = np.zeros(n_variants)  # koszt (cost)

    for i in range(n_variants):
        for j in range(n_criteria):
            if types[j] == 'profit':
                S_plus[i] += weighted[i, j]
            else:
                S_minus[i] += weighted[i, j]

    # Minimalna wartość S_minus
    S_minus_min = np.min(S_minus)

    # Obliczenie względnej ważności
    Q = np.zeros(n_variants)
    for i in range(n_variants):
        if S_minus[i] != 0:
            Q[i] = S_plus[i] + (S_minus_min * np.sum(S_minus)) / (S_minus[i] * np.sum(S_minus))
        else:
            Q[i] = S_plus[i] + (S_minus_min * np.sum(S_minus)) / (0.0001 * np.sum(S_minus))

    # Normalizacja wyników do przedziału [0, 1]
    if np.max(Q) - np.min(Q) != 0:
        Q_normalized = (Q - np.min(Q)) / (np.max(Q) - np.min(Q))
    else:
        Q_normalized = Q

    return Q_normalized


def electre_method(variants, criteria, matrix, concordance_threshold=0.6, discordance_threshold=0.4):
    """
    Metoda ELECTRE I (wersja uproszczona).
    Zwraca ranking oparty na relacji przewyższania.
    """
    X = np.array(matrix, dtype=float)
    weights = normalize_weights(criteria)
    types = [c['type'] for c in criteria]

    n_variants, n_criteria = X.shape

    # Normalizacja do przedziału [0, 1]
    norm_X = np.zeros_like(X)
    for j in range(n_criteria):
        col = X[:, j]
        if np.max(col) - np.min(col) != 0:
            norm_X[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col))
        else:
            norm_X[:, j] = col

    # Dla kryteriów koszt odwracamy wartości (bo mniejsze lepsze -> większe po odwróceniu)
    for j in range(n_criteria):
        if types[j] == 'cost':
            norm_X[:, j] = 1 - norm_X[:, j]

    # Macierz przewyższania
    dominance = np.zeros((n_variants, n_variants))

    for i in range(n_variants):
        for k in range(n_variants):
            if i == k:
                continue

            concordance_sum = 0
            discordance_present = False

            for j in range(n_criteria):
                if norm_X[i, j] >= norm_X[k, j]:
                    concordance_sum += weights[j]
                else:
                    # Sprawdzenie dyskordancji
                    diff = norm_X[k, j] - norm_X[i, j]
                    # Jeśli różnica jest duża, może to blokować przewyższanie
                    if diff > discordance_threshold:
                        discordance_present = True

            concordance = concordance_sum

            # Relacja przewyższania gdy concordance >= próg i brak dyskordancji
            if concordance >= concordance_threshold and not discordance_present:
                dominance[i, k] = 1

    # Obliczenie siły każdego wariantu (ile razy przewyższa innych)
    strength = np.sum(dominance, axis=1)
    weakness = np.sum(dominance, axis=0)

    # Wynik netto (strength - weakness) - im wyższy tym lepszy
    net_score = strength - weakness

    # Normalizacja do przedziału [0, 1] dla spójności z innymi metodami
    if np.max(net_score) - np.min(net_score) != 0:
        final_scores = (net_score - np.min(net_score)) / (np.max(net_score) - np.min(net_score))
    else:
        final_scores = np.ones(n_variants) * 0.5

    return final_scores


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compute', methods=['POST'])
def compute():
    data = request.get_json()
    variants = data['variants']
    criteria = data['criteria']
    matrix = data['matrix']
    method = data.get('method', 'saw')

    # Parametry dodatkowe dla metod
    vikor_v = data.get('vikor_v', 0.5)
    electre_concordance = data.get('electre_concordance', 0.6)
    electre_discordance = data.get('electre_discordance', 0.4)

    if not variants or not criteria or not matrix:
        return jsonify({'error': 'Brak danych'}), 400
    if len(variants) != len(matrix):
        return jsonify({'error': 'Liczba wariantów niezgodna z macierzą ocen'}), 400
    if len(criteria) != len(matrix[0]):
        return jsonify({'error': 'Liczba kryteriów niezgodna z macierzą ocen'}), 400

    for c in criteria:
        if c['weight'] < 0:
            return jsonify({'error': 'Wagi nie mogą być ujemne'}), 400

    try:
        _ = np.array(matrix, dtype=float)
    except:
        return jsonify({'error': 'Macierz ocen musi zawierać liczby'}), 400

    try:
        if method == 'saw':
            scores = saw_method(variants, criteria, matrix)
        elif method == 'topsis':
            scores = topsis_method(variants, criteria, matrix)
        elif method == 'vikor':
            scores = vikor_method(variants, criteria, matrix, vikor_v)
        elif method == 'copras':
            scores = copras_method(variants, criteria, matrix)
        elif method == 'electre':
            scores = electre_method(variants, criteria, matrix, electre_concordance, electre_discordance)
        else:
            return jsonify({'error': 'Nieznana metoda'}), 400

        sorted_indices = np.argsort(scores)[::-1]
        ranking = []
        for rank, idx in enumerate(sorted_indices, 1):
            ranking.append({
                'name': variants[idx],
                'score': round(float(scores[idx]), 4),
                'rank': rank
            })
        return jsonify({'ranking': ranking})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)