import re
import time
import warnings
from functools import wraps

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# простая токенизация: выделяем слова (буквы/цифры/подчёркивания)
_TOKEN = re.compile(r"\w+", flags=re.U)


def log_feature(func):
    """
    Декоратор для функций add_*_features.
    Логирует: имя функции, сколько времени заняло выполнение,
    и изменение формы DataFrame (n_rows, n_cols).
    """

    @wraps(func)
    def wrapper(df, *args, **kwargs):
        start_shape = df.shape
        start_time = time.time()
        print(f"[START] {func.__name__} | shape={start_shape}")

        result = func(df, *args, **kwargs)

        end_shape = result.shape
        elapsed = time.time() - start_time

        print(
            f"[END]   {func.__name__} | shape={end_shape} | Δcols={end_shape[1] - start_shape[1]} | time={elapsed:.2f}s"
        )
        print("-" * 60)
        return result

    return wrapper


def _has_tokens(texts):
    """Есть ли хотя бы один текст с хотя бы одним буквенно-цифровым символом."""
    return any(bool(re.search(r"\w", t)) for t in texts)


@log_feature
def add_tfidf_groupwise_cosine(
    df: pd.DataFrame,
    with_desc: bool = True,
) -> pd.DataFrame:
    """
    Для каждого query_id:
      1) фитим TfidfVectorizer на документах (title, опц. desc) этой группы,
      2) считаем косинусные сходства query vs doc в L2-нормированном TF-IDF пространстве,
      3) возвращаем колонки:
         - tfidf_cosine_title
         - (опц.) tfidf_cosine_desc
    """
    out = df.copy()

    n = len(out)
    out["tfidf_cosine_title"] = np.zeros(n, dtype=np.float32)
    if with_desc:
        out["tfidf_cosine_desc"] = np.zeros(n, dtype=np.float32)

    groups = out.groupby("query_id")
    for qid, idx in tqdm(
        groups.groups.items(), total=len(groups), desc="TFIDF per query"
    ):
        part = out.loc[idx]

        # текст запроса (один на группу)
        qtext = (
            str(part["query_text"].iloc[0])
            if pd.notna(part["query_text"].iloc[0])
            else ""
        )
        if not qtext.strip():
            # пустой запрос -> оставляем нули
            continue

        # Общая конфигурация векторайзера:
        # - token_pattern допускает односимвольные токены и кириллицу
        # - L2-нормировка, чтобы косинус = скалярному произведению
        # - dtype=float32, чтобы не раздувать память
        vec_kwargs = dict(
            ngram_range=(1, 2),
            lowercase=True,
            norm="l2",
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )

        # ---------- TITLE ----------
        titles = part["item_title"].fillna("").astype(str).tolist()
        if _has_tokens(titles):
            try:
                tfv_t = TfidfVectorizer(**vec_kwargs)
                Xd_t = tfv_t.fit_transform(titles)  # документы
                Xq_t = tfv_t.transform([qtext])  # запрос в том же пространстве
                sims_t = (Xd_t @ Xq_t.T).toarray().ravel().astype(np.float32)
                out.loc[idx, "tfidf_cosine_title"] = sims_t
            except ValueError as e:
                # Поймаем "empty vocabulary; perhaps the documents only contain stop words"
                warnings.warn(
                    f"[TFIDF title] qid={qid}: {e}. Группа пропущена (оставляем нули)."
                )

        # ---------- DESCRIPTION (опционально) ----------
        if with_desc:
            descs = part["item_description"].fillna("").astype(str).tolist()
            if _has_tokens(descs):
                try:
                    tfv_d = TfidfVectorizer(**vec_kwargs)
                    Xd_d = tfv_d.fit_transform(descs)
                    Xq_d = tfv_d.transform([qtext])
                    sims_d = (Xd_d @ Xq_d.T).toarray().ravel().astype(np.float32)
                    out.loc[idx, "tfidf_cosine_desc"] = sims_d
                except ValueError as e:
                    warnings.warn(
                        f"[TFIDF desc] qid={qid}: {e}. Группа пропущена (оставляем нули)."
                    )

    return out


def frac_query_in_text(q: pd.Series, t: pd.Series) -> np.ndarray:
    q_tok = q.fillna("").astype(str).str.lower().apply(lambda s: set(_TOKEN.findall(s)))
    t_tok = t.fillna("").astype(str).str.lower().apply(lambda s: set(_TOKEN.findall(s)))

    inter_len = []
    q_len = []
    for a, b in zip(q_tok.to_numpy(), t_tok.to_numpy()):
        inter_len.append(len(a.intersection(b)))
        q_len.append(max(len(a), 1))

    inter_len = np.asarray(inter_len, dtype=np.float32)
    q_len = np.asarray(q_len, dtype=np.float32)
    return inter_len / q_len


def fill_missing_cats(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s):
            # Переводим в строку + категорию
            out[c] = s.astype("Int64").astype("string").fillna("MISSING")
        else:
            out[c] = s.fillna("MISSING").replace("0", "MISSING").astype("string")
        out[c] = out[c].astype("category")
    return out


@log_feature
def add_price_features(df: pd.DataFrame, med_map: dict) -> pd.DataFrame:
    out = df.copy()
    # 1) логарифм цены — сглаживаем «очень большие» значения
    out["price_log"] = np.log1p(out["price"].fillna(0)).astype("float32")
    # 2) нормализация цены относительно медианы по категории
    out["price_median_cat"] = (
        out["item_cat_id"].astype("object").map(med_map).astype("float32")
    )
    out["price_norm"] = (
        (out["price"] / out["price_median_cat"]).fillna(1.0).astype("float32")
    )
    out["price_norm"] = out["price_norm"].replace([np.inf, -np.inf], 1.0)
    # 3) бесплатное объявление
    out["is_free"] = (out["price"].fillna(0) == 0).astype("int8")
    return out.drop(columns=["price_median_cat"])


@log_feature
def add_ctr_features(df: pd.DataFrame, clip_val: float) -> pd.DataFrame:
    out = df.copy()
    x = out["item_query_click_conv"]
    out["click_conv"] = x.fillna(0).clip(lower=0, upper=clip_val).astype("float32")
    return out


@log_feature
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["query_text", "item_title", "item_description"]:
        s = out[col].fillna("")
        out[col + "_len_chars"] = s.str.len().astype("int32")
        out[col + "_len_words"] = s.str.count(r"\S+").astype("int32")
    # новые фичи: доля слов запроса, найденных в title/description
    out["q_in_title_frac"] = frac_query_in_text(out["query_text"], out["item_title"])
    out["q_in_desc_frac"] = frac_query_in_text(
        out["query_text"], out["item_description"]
    )
    return out


@log_feature
def add_match_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # перед сравнением приведём к объектам (строкам), чтобы не падать на несовпадающих категориях
    out["match_cat"] = (
        out["query_cat"].astype("object") == out["item_cat_id"].astype("object")
    ).astype("int8")
    out["match_mcat"] = (
        out["query_mcat"].astype("object") == out["item_mcat_id"].astype("object")
    ).astype("int8")
    out["match_loc"] = (
        out["query_loc"].astype("object") == out["item_loc"].astype("object")
    ).astype("int8")
    return out


@log_feature
def add_price_in_query_feats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["price_rank_in_query"] = (
        out.groupby("query_id")["price"].rank(method="average", pct=True)
    ).astype("float32")
    return out


def add_freq_enc(
    df_train: pd.DataFrame, df_test: pd.DataFrame, cols
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Частоты значений считаем по train и применяем к обеим таблицам.
    Это дешёвый способ дать модели «понимание популярности».
    """
    tr = df_train.copy()
    te = df_test.copy()
    for c in cols:
        freq = tr[c].astype("object").value_counts(normalize=True)
        tr[c + "_freq"] = tr[c].astype("object").map(freq).astype("float32")
        te[c + "_freq"] = te[c].astype("object").map(freq).astype("float32").fillna(0.0)
    return tr, te


@log_feature
def add_price_z_in_query(df):
    g = df.groupby("query_id")["price"]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    df["price_z_in_query"] = ((df["price"] - mu) / sd).fillna(0).astype("float32")
    return df


@log_feature
def tweak_click_conv(df, eps=1e-6):
    df["click_conv_log"] = np.log1p(df["click_conv"] + eps).astype("float32")
    # нормализация внутри запроса
    g = df.groupby("query_id")["click_conv_log"]
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    df["click_conv_log_z_in_query"] = (
        ((df["click_conv_log"] - mu) / sd).fillna(0).astype("float32")
    )
    return df


def add_2d_freq(train_df, test_df, col_a, col_b, out):
    freq = (
        train_df[[col_a, col_b]]
        .astype("object")
        .value_counts(normalize=True)
        .rename(out)
    )
    for df in (train_df, test_df):
        df[out] = freq.reindex(
            list(zip(df[col_a].astype("object"), df[col_b].astype("object")))
        ).to_numpy(dtype="float32")
        df[out] = df[out].fillna(0.0).astype("float32")
    return train_df, test_df


@log_feature
def add_initial_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Предполагаем, что df уже отсортирован как вернул исходный ретривер
    внутри каждой группы query_id. Добавляем позиционные фичи.
    """
    out = df.copy()
    r = (out.groupby("query_id").cumcount() + 1).astype("int32")
    out["initial_rank"] = r

    # размер группы
    grp_size = out.groupby("query_id")["initial_rank"].transform("max").clip(lower=1)

    # простые нормировки
    out["initial_rank_inv"] = (1.0 / (r + 1)).astype("float32")
    out["initial_rank_pct"] = (1.0 - (r - 1) / (grp_size - 1).replace(0, 1)).astype(
        "float32"
    )

    # флаги топ-k
    out["init_in_top1"] = (r <= 1).astype("int8")
    out["init_in_top3"] = (r <= 3).astype("int8")
    out["init_in_top10"] = (r <= 10).astype("int8")

    return out


@log_feature
def add_group_norms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z и min-max нормировки внутри query_id для готовых колонок:
    tfidf_cosine_title, tfidf_cosine_desc, click_conv.
    """
    out = df.copy()
    cols = ["tfidf_cosine_title", "tfidf_cosine_desc", "click_conv_log"]
    for c in cols:
        g = out.groupby("query_id")[c]
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        out[c + "_z_in_q"] = ((out[c] - mu) / sd).fillna(0).astype("float32")

        mn = g.transform("min")
        mx = g.transform("max")
        denom = (mx - mn).replace(0, np.nan)
        out[c + "_mm_in_q"] = ((out[c] - mn) / denom).fillna(0).astype("float32")
    return out


@log_feature
def add_basic_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Простые интеракции с позицией (initial_rank_inv / topk).
    """
    out = df.copy()

    # tfidf * inv-rank
    out["tfidf_title_x_init"] = (
        out["tfidf_cosine_title"].astype("float32")
        * out["initial_rank_inv"].astype("float32")
    ).astype("float32")

    # price_rank * inv-rank
    out["price_rank_x_init"] = (
        out["price_rank_in_query"].astype("float32")
        * out["initial_rank_inv"].astype("float32")
    ).astype("float32")

    # ctr_z * inv-rank
    out["ctr_z_x_init"] = (
        out["click_conv_log_z_in_query"].astype("float32")
        * out["initial_rank_inv"].astype("float32")
    ).astype("float32")

    # top3 × match_cat
    out["top3_x_match_cat"] = (
        out["init_in_top3"].astype("int8") * out["match_cat"].astype("int8")
    ).astype("int8")

    return out


@log_feature
def add_tfidf_rank(df: pd.DataFrame, tfidf_col: str) -> pd.DataFrame:
    out = df.copy()
    out["tfidf_rank_in_query"] = (
        out.groupby("query_id")[tfidf_col]
        .rank(method="first", ascending=False, pct=True)
        .astype("float32")
    )
    return out


# Нормализация юникода для м^2 и т.п.: "²" -> "2"
_SUP2 = "\N{SUPERSCRIPT TWO}"

_NUM_RE = re.compile(r"\d+")
# Список поддерживаемых единиц (кириллица+латиница), нормализуем к канону
_UNIT_CANON = {
    # длина
    "мм": "mm",
    "mm": "mm",
    "см": "cm",
    "cm": "cm",
    "м": "m",
    "m": "m",
    # площадь
    f"м{_SUP2}": "m2",
    "м2": "m2",
    "m2": "m2",
    f"m{_SUP2}": "m2",
    "см2": "cm2",
    "cm2": "cm2",
    # масса
    "кг": "kg",
    "kg": "kg",
    "г": "g",
    "g": "g",
    "мг": "mg",
    "mg": "mg",
    # объём
    "л": "l",
    "l": "l",
    "мл": "ml",
    "ml": "ml",
    # ёмкость/память
    "гб": "gb",
    "gb": "gb",
    "тб": "tb",
    "tb": "tb",
    "мб": "mb",
    "mb": "mb",
}

# Регекс под единицы: ищем как отдельные токены и «слепленные» с числами (например, 128gb)
_UNIT_PATTERN = re.compile(
    r"(?i)\b(\d+)?\s*(" + "|".join(map(re.escape, _UNIT_CANON.keys())) + r")\b"
)


def _norm_text_units(s: str) -> str:
    # нормализуем "²" -> "2" и приводим к нижнему регистру
    return (s or "").replace(_SUP2, "2").lower()


def _extract_numbers(s: str):
    s = _norm_text_units(s)
    return set(_NUM_RE.findall(s))


def _extract_units(s: str):
    s = _norm_text_units(s)
    units = set()
    for m in _UNIT_PATTERN.finditer(s):
        raw = m.group(2).lower()
        canon = _UNIT_CANON.get(raw)
        if canon:
            units.add(canon)
    return units


@log_feature
def add_numeric_and_unit_matches(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Признаки:
      - has_num_in_query: в query есть числа
      - num_match_in_title/desc: пересечение чисел query и title/desc
      - unit_match_in_title/desc: пересечение единиц измерения query и title/desc
      - num_or_unit_match_any: любое из совпадений в title или desc
      - num_and_unit_both_match_any: и число, и единица совпали хотя бы где-то (title/desc)
    """
    out = df.copy()

    q_nums = df["query_text"].astype(str).map(_extract_numbers)
    q_units = df["query_text"].astype(str).map(_extract_units)

    t_nums = df["item_title"].astype(str).map(_extract_numbers)
    t_units = df["item_title"].astype(str).map(_extract_units)

    d_nums = df["item_description"].astype(str).map(_extract_numbers)
    d_units = df["item_description"].astype(str).map(_extract_units)

    # Бинарки (int8 экономит память)
    out["has_num_in_query"] = (q_nums.map(len) > 0).astype("int8")

    out["num_match_in_title"] = q_nums.combine(
        t_nums, lambda a, b: len(a & b) > 0
    ).astype("int8")
    out["num_match_in_desc"] = q_nums.combine(
        d_nums, lambda a, b: len(a & b) > 0
    ).astype("int8")

    out["unit_match_in_title"] = q_units.combine(
        t_units, lambda a, b: len(a & b) > 0
    ).astype("int8")
    out["unit_match_in_desc"] = q_units.combine(
        d_units, lambda a, b: len(a & b) > 0
    ).astype("int8")

    out["num_or_unit_match_any"] = (
        out["num_match_in_title"]
        | out["num_match_in_desc"]
        | out["unit_match_in_title"]
        | out["unit_match_in_desc"]
    ).astype("int8")

    out["num_and_unit_both_match_any"] = (
        (out["num_match_in_title"] | out["num_match_in_desc"])
        & (out["unit_match_in_title"] | out["unit_match_in_desc"])
    ).astype("int8")

    return out


@log_feature
def add_price_bins_by_category(
    df: pd.DataFrame,
    cat_col: str = "item_cat_id",
    price_col: str = "price",
) -> pd.DataFrame:
    """
    Бинарные фичи по цене в разрезе категории:
      - price_is_zero
      - price_lt_cat_median
      - price_gt_cat_p90
    """
    out = df.copy()
    p = out[price_col].astype(float)

    # Групповые квантили
    grp = out.groupby(cat_col, dropna=False)
    cat_median = grp[price_col].transform("median")
    cat_p90 = grp[price_col].transform(lambda s: np.nanpercentile(s, 90))

    out["price_is_zero"] = (p.fillna(0.0) == 0.0).astype("int8")
    out["price_lt_cat_median"] = (p < cat_median).fillna(0).astype("int8")
    out["price_gt_cat_p90"] = (p > cat_p90).fillna(0).astype("int8")

    return out


@log_feature
def add_price_over_median_cat_loc(
    df: pd.DataFrame,
    cat_col: str = "item_cat_id",
    loc_col: str = "item_loc",
    price_col: str = "price",
    clip_min: float = 0.0,
    clip_max: float | None = None,
) -> pd.DataFrame:
    """
    Фича:
      - price_over_median_cat_loc = price / median(price | category, location)

    Безопасности:
      - если медиана == 0 или NaN → ставим 1.0 (нейтрально)
      - клипуем по [clip_min, clip_max] (если задано), dtype=float32
    """
    out = df.copy()
    p = out[price_col].astype(float)

    grp = out.groupby([cat_col, loc_col], dropna=False)
    med = grp[price_col].transform("median")

    denom = med.replace(0.0, np.nan)
    ratio = p / denom
    ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)  # нейтральное значение

    if clip_max is not None:
        ratio = ratio.clip(lower=clip_min, upper=clip_max)
    elif clip_min > 0:
        ratio = ratio.clip(lower=clip_min)

    out["price_over_median_cat_loc"] = ratio.astype("float32")
    return out
