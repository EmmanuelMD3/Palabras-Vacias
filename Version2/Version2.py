import os
import re
import spacy
from collections import Counter
import pandas as pd
from itertools import product
from tqdm import tqdm
from datetime import datetime


"""
Versión del método del artículo:
"Stop-words in Keyphrase Extraction Problem"

MISMA ESTRUCTURA QUE TU CÓDIGO, con mejoras mínimas:
- Filtra tokens basura (%,-, H/sub, tokens sin letras)
- Usa lemma_ para mejorar matching
- Normaliza GOLD y SYS igual (para que coincidan)
- En train-based: log CSV con ACEPTADOS primero y luego RECHAZADOS
- Mantiene greedy: prueba 1 por 1, actualiza F_base al aceptar
"""


# --------- CONFIGURACIÓN ---------
RESUMENES_DIR = r"C:\Users\chemo\OneDrive\Documentos\Servicio\Servicio Social\resumenes_train"
KEYS_DIR = r"C:\Users\chemo\OneDrive\Documentos\Servicio\Servicio Social\frases_clave_train"

SPACY_MODEL = "en_core_web_sm"

# Parámetros método frecuencia
TOP_N_STOP_CANDIDATES = 200
FREQ_STEPS = [50, 100, 200]
NONSTOP_STEPS = [0, 50, 100]

# Parámetros train-based
TRAIN_METHOD_A = 0.0001
MAX_TRAIN_CAND = 1000

# Salidas
CSV_OUT = "results_advanced_method.csv"
TRAIN_LOG = "train_decisions_log.csv"


# --------- spaCy ---------
nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])


# --------- NORMALIZACIÓN (MEJORA) ---------
def normalize_phrase(s: str) -> str:
    """
    Normaliza frases para que GOLD y SYS coincidan mejor:
    - lower
    - deja letras/números/espacios/guion
    - colapsa espacios
    """
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)   # quita símbolos raros
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_garbage_token(w: str) -> bool:
    """
    Detecta tokens basura:
    - sin letras (%, 123, etc.)
    - con slash (H/sub)
    - guiones sueltos
    """
    if not w:
        return True
    if "/" in w:
        return True
    if w in {"-", "--", "—"}:
        return True
    if not re.search(r"[a-z]", w):
        return True
    return False


# --------- UTILIDADES DE LECTURA ---------
def leer_carpeta_txt(ruta):
    archivos = {}
    for fname in sorted(os.listdir(ruta)):
        if fname.lower().endswith(".txt"):
            with open(os.path.join(ruta, fname), "r", encoding="utf-8", errors="ignore") as f:
                archivos[fname] = f.read().strip()
    return archivos


def leer_gold_per_doc(keys_dir):
    """
    {archivo: [frases_clave_normalizadas]}
    """
    gold = {}
    for fname, text in leer_carpeta_txt(keys_dir).items():
        frases = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            frases.append(normalize_phrase(line))
        gold[fname] = frases
    return gold


# --------- PREPROCESAMIENTO spaCy ---------
def preprocess_docs(resumenes):
    print("Preprocesando spaCy en batch...")
    docs = list(nlp.pipe(
        resumenes.values(),
        batch_size=32,
        n_process=os.cpu_count() or 1
    ))
    return dict(zip(resumenes.keys(), docs))


# --------- EXTRACCIÓN DE CANDIDATOS (MISMA IDEA + FILTRO + LEMMA + NORMALIZE) ---------
def extract_candidates_from_doc(doc, stopset):
    """
    MISMA LÓGICA BASE:
    Secuencias de ADJ/NOUN (agrego PROPN por si acaso)
    Longitud >= 2

    MEJORAS:
    - usa lemma_
    - filtra basura (%,-, H/sub, tokens sin letras)
    - normaliza frase final
    """
    out = []
    span = []

    for tok in doc:
        raw = tok.text.lower().strip()
        lem = tok.lemma_.lower().strip()

        ok = (
            tok.pos_ in ("NOUN", "ADJ", "PROPN")
            and not tok.is_punct
            and not tok.is_space
            and raw
            and lem
            and raw not in stopset
            and lem not in stopset
            and len(lem) > 1
            and not is_garbage_token(raw)
            and not is_garbage_token(lem)
        )

        if ok:
            span.append(lem)
        else:
            if len(span) > 1:
                out.append(normalize_phrase(" ".join(span)))
            span = []

    if len(span) > 1:
        out.append(normalize_phrase(" ".join(span)))

    return out


def extract_all(pre_docs, stopset):
    all_sys = []
    for doc in pre_docs.values():
        all_sys.extend(extract_candidates_from_doc(doc, stopset))
    return all_sys


# --------- MÉTRICAS (tu versión global por set, solo normalizando) ---------
def precision_recall_f(gold_list, sys_list):
    """
    Calcula Precision, Recall y F-score globales.
    Nota: usa sets como tú lo tenías.
    """
    gold = set(normalize_phrase(x) for x in gold_list)
    sys = set(normalize_phrase(x) for x in sys_list)

    if not sys and not gold:
        return 1.0, 1.0, 1.0

    tp = len(gold & sys)
    prec = tp / len(sys) if sys else 0.0
    rec = tp / len(gold) if gold else 0.0
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f


# --------- FRECUENCIAS Y H-POINT ---------
def build_frequency_dicts(pre_docs):
    """
    MISMA FUNCIÓN, pero con mejora:
    - frecuencias sobre lemma
    - filtra basura
    """
    corpus_freq = Counter()
    candidate_token_freq = Counter()

    for doc in pre_docs.values():

        # Frecuencia global de tokens
        for tok in doc:
            if tok.pos_ in ("NOUN", "ADJ", "PROPN"):
                w = tok.lemma_.lower().strip()
                if len(w) > 1 and not is_garbage_token(w) and re.fullmatch(r"[a-z][a-z\-]*", w):
                    corpus_freq[w] += 1

        # Frecuencia dentro de frases candidatas
        cands = extract_candidates_from_doc(doc, set())
        for phrase in cands:
            for token in phrase.split():
                if len(token) > 1 and not is_garbage_token(token) and re.fullmatch(r"[a-z][a-z\-]*", token):
                    candidate_token_freq[token] += 1

    return corpus_freq, candidate_token_freq


def compute_h_point(freq_counter):
    freqs = sorted(freq_counter.values(), reverse=True)
    h = 0
    for i, f in enumerate(freqs, start=1):
        if f >= i:
            h = i
        else:
            break
    return h


# --------- MÉTODO 1: FRECUENCIA + COMPENSACIÓN ---------
def frequency_based_method(pre_docs, gold_all):

    corpus_freq, cand_token_freq = build_frequency_dicts(pre_docs)

    corpus_sorted = [w for w, _ in corpus_freq.most_common(TOP_N_STOP_CANDIDATES)]
    cand_sorted = [w for w, _ in cand_token_freq.most_common(TOP_N_STOP_CANDIDATES)]

    print(f"[INFO] h-point: {compute_h_point(cand_token_freq)}")

    results = []

    for stop_k, non_k in product(FREQ_STEPS, NONSTOP_STEPS):

        stop_candidates = set(corpus_sorted[:stop_k])
        nonstop_candidates = set(cand_sorted[:non_k])

        final_stoplist = stop_candidates - nonstop_candidates

        sys_phrases = extract_all(pre_docs, final_stoplist)
        prec, rec, f = precision_recall_f(gold_all, sys_phrases)

        print(f"[FREQ] stop_k={stop_k} non_k={non_k} f={f:.4f}")

        results.append({
            "method": "frequency_compensation",
            "stop_k": stop_k,
            "non_k": non_k,
            "stop_count": len(final_stoplist),
            "precision": prec,
            "recall": rec,
            "f_score": f
        })

    return results


# --------- MÉTODO 2: TRAIN-BASED (TU ESTRUCTURA + MEJORAS + LOG ORDENADO) ---------
def train_based_method(pre_docs, gold_all):
    """
    MISMO greedy que tú:
    - candidatos por frecuencia
    - prueba token por token
    - si mejora > A, acepta y actualiza F_base
    - registra aceptados y rechazados
    - guarda CSV: primero aceptados, luego rechazados
    """

    corpus_freq, _ = build_frequency_dicts(pre_docs)

    # candidatos limpios (ya vienen filtrados en build_frequency_dicts)
    candidates = [w for w, _ in corpus_freq.most_common(MAX_TRAIN_CAND)]

    # baseline
    baseline_sys = extract_all(pre_docs, set())
    _, _, f_base = precision_recall_f(gold_all, baseline_sys)
    print(f"[TRAIN] Baseline F={f_base:.4f}")

    selected_stopwords = set()
    accepted_tokens = []
    rejected_tokens = []

    for token in tqdm(candidates, desc="Train-based"):
        trial_stop = selected_stopwords | {token}
        sys_phrases = extract_all(pre_docs, trial_stop)
        _, _, f_trial = precision_recall_f(gold_all, sys_phrases)

        improvement = f_trial - f_base

        if improvement > TRAIN_METHOD_A:
            selected_stopwords.add(token)
            f_base = f_trial

            accepted_tokens.append((token, improvement, f_base))
            print(f"[+] {token} accepted → ΔF={improvement:.6f} → F={f_base:.4f}")
        else:
            rejected_tokens.append((token, improvement))

    # Resultado final (con stopwords finales)
    sys_final = extract_all(pre_docs, selected_stopwords)
    prec, rec, f_final = precision_recall_f(gold_all, sys_final)

    print(f"\n[TRAIN] Final stopwords={len(selected_stopwords)} F={f_final:.4f}")

    # ---- Guardar CSV (ACEPTADOS primero y luego RECHAZADOS, como tú quieres) ----
    rows = (
        [(t, "accepted", imp, f) for t, imp, f in accepted_tokens] +
        [(t, "rejected", imp, None) for t, imp in rejected_tokens]
    )
    df_train_log = pd.DataFrame(rows, columns=["token", "status", "delta_f", "f_score_after"])

    # intenta guardar con nombre normal; si Excel lo tiene abierto, usa timestamp
    try:
        df_train_log.to_csv(TRAIN_LOG, index=False, encoding="utf-8")
        print("Log guardado en:", TRAIN_LOG)
    except PermissionError:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"train_decisions_log_{ts}.csv"
        df_train_log.to_csv(alt, index=False, encoding="utf-8")
        print("No se pudo escribir train_decisions_log.csv (probablemente abierto).")
        print("Log guardado en:", alt)

    return {
        "method": "train_based",
        "stopwords": selected_stopwords,
        "precision": prec,
        "recall": rec,
        "f_score": f_final
    }


# --------- FUNCIÓN PRINCIPAL ---------
def main():

    resumenes = leer_carpeta_txt(RESUMENES_DIR)
    gold = leer_gold_per_doc(KEYS_DIR)

    # usar solo archivos coincidentes
    inter = sorted(set(resumenes) & set(gold))
    resumenes = {k: resumenes[k] for k in inter}
    gold = {k: gold[k] for k in inter}

    # gold global
    gold_all = []
    for g in gold.values():
        gold_all.extend(g)

    # Preprocesamiento spaCy
    pre_docs = preprocess_docs(resumenes)

    # Baseline sin stoplist
    baseline_sys = extract_all(pre_docs, set())
    p0, r0, f0 = precision_recall_f(gold_all, baseline_sys)
    print(f"Baseline → P={p0:.4f} R={r0:.4f} F={f0:.4f}")

    # Método frecuencia
    freq_results = frequency_based_method(pre_docs, gold_all)

    # Método train-based (TU MÉTODO, solo mejorado)
    train_result = train_based_method(pre_docs, gold_all)

    # Guardar resultados
    rows = freq_results + [{
        "method": "train_based",
        "stop_k": None,
        "non_k": None,
        "stop_count": len(train_result["stopwords"]),
        "precision": train_result["precision"],
        "recall": train_result["recall"],
        "f_score": train_result["f_score"]
    }]

    df = pd.DataFrame(rows)
    df.to_csv(CSV_OUT, index=False, encoding="utf-8")

    print("\nTop resultados:")
    print(df.sort_values("f_score", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()