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
    X = np.array(matrix, dtype=float)
    weights = normalize_weights(criteria)
    types = [c['type'] for c in criteria]

    norm_X = X / np.sqrt(np.sum(X**2, axis=0))
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

    dist_to_ideal = np.sqrt(np.sum((weighted - ideal)**2, axis=1))
    dist_to_anti = np.sqrt(np.sum((weighted - anti_ideal)**2, axis=1))

    scores = dist_to_anti / (dist_to_ideal + dist_to_anti)
    return scores

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