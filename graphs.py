"""
graphs.py — Анализ инвестиционных результатов IMOEX и S&P500
Запуск: python graphs.py
Предполагается, что рядом лежат файлы IMOEX.csv и s-and-p-500.csv

Выводит текстовый лог всех ключевых данных в stdout —
его можно скопировать и отправить в LLM на анализ.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# ─── НАСТРОЙКИ ───────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

# Настройки генерации графиков
FIGURE_DPI = 300                 # Разрешение картинок (300 для печати, 100-130 для экрана)
ENABLE_WATERMARK = False          # Включить/выключить подпись автора на графиках
WATERMARK_TEXT = "Михаил Шардин https://shardin.name/"

# Настройки тепловой карты (адаптация для цветовосприятия)
# RdBu_r   — (Рекомендуется) Красный (убыток) -> Белый -> Синий (прибыль). Идеально различается.
# coolwarm — Похож на RdBu_r, но цвета чуть более мягкие/пастельные.
# cividis  — Специально создана для людей с любыми формами дальтонизма (Темно-синий -> Желтый).
HEATMAP_CMAP = "RdBu_r" 
# Порог линейной зоны (%). Чем он выше, тем лучше детализированы "средние" доходы до перехода в лог. шкалу
HEATMAP_LINTHRESH = 30 

# Настройки горизонтов инвестирования
HORIZONS = [1, 3, 5, 10, 15, 20]
HORIZON_COLORS = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#b07aa1"]

# Настройки шрифта (чтобы кириллица отображалась корректно)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": FIGURE_DPI,
})

# ─── Кризисные даты ───────────────────────────────────────────────────────────

SP500_CRISES = [
    ("1907\nPanic",        1907,  1, "#8B0000", "Panic of 1907: падение ~40-50%"),
    ("1929\nCrash",        1929, 10, "#CC0000", "Wall Street Crash 1929: падение ~86%, восстановление — десятилетия"),
    ("1937\nRecession",    1937,  3, "#FF4444", "Рецессия 1937-38 внутри Великой депрессии: просадка ~50%"),
    ("1973\nOil",          1973, 10, "#FF8800", "Нефтяной кризис 1973-74: просадка ~48%"),
    ("1987\nBlack Mon",    1987, 10, "#FFAA00", "Black Monday 1987: однодневное падение ~22%"),
    ("2000\nDot-com",      2000,  3, "#886600", "Крах доткомов 2000-02: просадка S&P500 ~49%"),
    ("2008\nGFC",          2008,  9, "#000088", "Global Financial Crisis 2007-09: просадка ~57%"),
    ("2020\nCOVID",        2020,  2, "#008800", "COVID crash 2020: падение ~34% за несколько недель"),
    ("2022\nFed",          2022,  1, "#006666", "Медвежий рынок 2022 из-за роста ставок ФРС: просадка ~25%"),
]

IMOEX_CRISES = [
    ("1998\nДефолт",       1998,  8, "#8B0000", "Российский финансовый кризис 1998: один из сильнейших ударов по активам"),
    ("2000\nДоткомы",      2000,  3, "#FF4444", "Крах доткомов 2000-02: затронул Россию слабее, но снижение было заметным"),
    ("2008\nГФК",          2008,  9, "#000088", "Global Financial Crisis 2008-09"),
    ("2011\nЕврокризис",   2011,  8, "#FF8800", "Европейский долговой кризис 2011-12 и замедление роста"),
    ("2014\nСанкции",      2014,  3, "#886600", "Падение нефти и валютный кризис 2014-15"),
    ("2020\nCOVID",        2020,  2, "#008800", "Пандемия COVID-19 2020"),
    ("2022\nСВО",          2022,  2, "#9900CC", "СВО 2022"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════════════════════

def add_watermark(fig):
    """Добавляет водяной знак (копирайт) на график, если он включен в настройках."""
    if ENABLE_WATERMARK:
        fig.text(0.99, 0.015, WATERMARK_TEXT, 
                 fontsize=9, color="gray", alpha=0.6, 
                 ha="right", va="bottom", transform=fig.transFigure)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ЗАГРУЗКА ДАННЫХ
# ═══════════════════════════════════════════════════════════════════════════════

def load_imoex(path="IMOEX.csv"):
    df = pd.read_csv(path, sep=";", decimal=",", thousands=None, low_memory=False)
    df = df[["TRADEDATE", "CLOSE"]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["close"] = pd.to_numeric(df["close"].astype(str).str.replace(",", "."), errors="coerce")
    df = df.dropna().sort_values("date").reset_index(drop=True)
    df = df.set_index("date")
    monthly = df["close"].resample("ME").last().dropna()
    return monthly

def load_sp500(path="s-and-p-500.csv"):
    df = pd.read_csv(path, low_memory=False)
    df = df[["Date", "SP500"]].copy()
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().sort_values("date").set_index("date")
    monthly = df["close"].resample("ME").last().dropna()
    return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# 2. РАСЧЁТ ДОХОДНОСТЕЙ
# ═══════════════════════════════════════════════════════════════════════════════

def calc_returns(monthly: pd.Series, horizons=HORIZONS):
    """Для каждой даты входа считаем полную доходность и CAGR на каждый горизонт."""
    results = {}
    for h in horizons:
        months = h * 12
        if months >= len(monthly):
            results[h] = pd.DataFrame({"total_ret": [], "cagr": []})
            continue
        start = monthly.iloc[:-months]
        end   = monthly.iloc[months:]
        end   = end.set_axis(start.index)
        total_ret = (end.values / start.values) - 1
        cagr      = (end.values / start.values) ** (1 / h) - 1
        results[h] = pd.DataFrame({"total_ret": total_ret, "cagr": cagr}, index=start.index)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ТЕКСТОВЫЙ ЛОГ ДЛЯ LLM
# ═══════════════════════════════════════════════════════════════════════════════

def log_separator(title=""):
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")
    else:
        print("-" * 70)

def log_data(sp500, imoex, ret_sp, ret_im):
    """Выводит все ключевые данные в текстовом виде для передачи в LLM."""

    log_separator("ИСХОДНЫЕ ДАННЫЕ")
    print(f"S&P500:  с {sp500.index[0].date()} по {sp500.index[-1].date()}  ({len(sp500)} мес.)")
    print(f"  Первое значение: {sp500.iloc[0]:.2f}")
    print(f"  Последнее значение: {sp500.iloc[-1]:.2f}")
    print(f"  Общий рост за всё время: {(sp500.iloc[-1]/sp500.iloc[0]-1)*100:.1f}%")
    print()
    print(f"IMOEX:   с {imoex.index[0].date()} по {imoex.index[-1].date()}  ({len(imoex)} мес.)")
    print(f"  Первое значение: {imoex.iloc[0]:.2f}")
    print(f"  Последнее значение: {imoex.iloc[-1]:.2f}")
    print(f"  Общий рост за всё время: {(imoex.iloc[-1]/imoex.iloc[0]-1)*100:.1f}%")

    log_separator("РАСПРЕДЕЛЕНИЕ ИТОГОВОЙ ДОХОДНОСТИ ПО ГОРИЗОНТАМ")
    for name, ret_dict in [("S&P500 (USD)", ret_sp), ("IMOEX (RUB)", ret_im)]:
        print(f"\n{name}")
        print(f"{'Горизонт':>10} {'N':>6} {'Медиана%':>10} {'Среднее%':>10} "
              f"{'P10%':>8} {'P25%':>8} {'P75%':>8} {'P90%':>8} "
              f"{'Мин%':>8} {'Макс%':>8} {'P(убыток)':>10}")
        print("-" * 100)
        for h in HORIZONS:
            r = ret_dict[h]["total_ret"].dropna() * 100
            if len(r) == 0:
                print(f"{h:>8}л {'нет данных':>6}")
                continue
            print(f"{h:>8}л "
                  f"{len(r):>6} "
                  f"{r.median():>10.1f} "
                  f"{r.mean():>10.1f} "
                  f"{r.quantile(0.10):>8.1f} "
                  f"{r.quantile(0.25):>8.1f} "
                  f"{r.quantile(0.75):>8.1f} "
                  f"{r.quantile(0.90):>8.1f} "
                  f"{r.min():>8.1f} "
                  f"{r.max():>8.1f} "
                  f"{(r<0).mean()*100:>9.1f}%")

    log_separator("CAGR (СРЕДНЕГОДОВАЯ ДОХОДНОСТЬ) ПО ГОРИЗОНТАМ")
    for name, ret_dict in [("S&P500 (USD)", ret_sp), ("IMOEX (RUB)", ret_im)]:
        print(f"\n{name}")
        print(f"{'Горизонт':>10} {'Медиана CAGR%':>15} {'Среднее CAGR%':>15} "
              f"{'Худший CAGR%':>14} {'Лучший CAGR%':>14}")
        print("-" * 72)
        for h in HORIZONS:
            r = ret_dict[h]["cagr"].dropna() * 100
            if len(r) == 0:
                continue
            print(f"{h:>8}л "
                  f"{r.median():>15.2f} "
                  f"{r.mean():>15.2f} "
                  f"{r.min():>14.2f} "
                  f"{r.max():>14.2f}")

    log_separator("ДОХОДНОСТЬ ПО ДЕСЯТИЛЕТИЯМ (горизонт 10 лет, CAGR %)")
    for name, ret_dict in [("S&P500 (USD)", ret_sp), ("IMOEX (RUB)", ret_im)]:
        h = 10
        r = ret_dict[h]["cagr"].dropna() * 100
        if len(r) == 0:
            print(f"\n{name}: недостаточно данных")
            continue
        df = r.to_frame("cagr")
        df["decade"] = (df.index.year // 10) * 10
        tbl = df.groupby("decade")["cagr"].agg(
            N="count", median="median", worst="min", best="max", mean="mean"
        )
        print(f"\n{name}")
        print(f"{'Десятилетие':>14} {'N':>5} {'Медиана%':>10} {'Среднее%':>10} "
              f"{'Худший%':>9} {'Лучший%':>9}")
        print("-" * 60)
        for decade, row in tbl.iterrows():
            print(f"{str(decade)+'-е':>14} "
                  f"{int(row['N']):>5} "
                  f"{row['median']:>10.2f} "
                  f"{row['mean']:>10.2f} "
                  f"{row['worst']:>9.2f} "
                  f"{row['best']:>9.2f}")

    log_separator("ВЕРОЯТНОСТЬ УБЫТКА P(R < 0) ПО ГОРИЗОНТАМ")
    print(f"\n{'Горизонт':>10}   {'S&P500':>8}   {'IMOEX':>8}")
    print("-" * 34)
    for h in HORIZONS:
        r_sp = ret_sp[h]["total_ret"].dropna()
        r_im = ret_im[h]["total_ret"].dropna()
        p_sp = (r_sp < 0).mean() * 100 if len(r_sp) else float("nan")
        p_im = (r_im < 0).mean() * 100 if len(r_im) else float("nan")
        print(f"{h:>8}л   {p_sp:>7.1f}%   {p_im:>7.1f}%")

    log_separator("СРАВНЕНИЕ ДОХОДНОСТИ ВОКРУГ КРИЗИСНЫХ ДАТ")
    _log_crisis_returns("S&P500 (USD)", sp500, ret_sp, SP500_CRISES)
    _log_crisis_returns("IMOEX  (RUB)", imoex, ret_im, IMOEX_CRISES)

    log_separator("КОНЕЦ ЛОГА")

def _log_crisis_returns(name, monthly, ret_dict, crises):
    """Для каждого кризиса показывает доходность при входе за 3 года до и после."""
    print(f"\n{name}")
    for label, year, month, color, desc in crises:
        crisis_date = pd.Timestamp(year=year, month=month, day=1)
        if crisis_date < monthly.index[0] or crisis_date > monthly.index[-1]:
            continue
        print(f"\n  [{year}/{month:02d}] {desc}")
        # Входы: -36, -24, -12, 0, +12, +24, +36 месяцев относительно кризиса
        offsets = [-36, -24, -12, 0, 12, 24, 36]
        for h in [1, 3, 5, 10]:
            row_parts = [f"    горизонт {h:2d}л: "]
            for off in offsets:
                entry = crisis_date + pd.DateOffset(months=off)
                # Ближайшая дата в данных
                idx = monthly.index.searchsorted(entry)
                if idx >= len(monthly):
                    row_parts.append(f"off={off:+4d}м:    н/д  ")
                    continue
                entry_price = monthly.iloc[idx]
                exit_idx = idx + h * 12
                if exit_idx >= len(monthly):
                    row_parts.append(f"off={off:+4d}м:    н/д  ")
                    continue
                exit_price = monthly.iloc[exit_idx]
                total = (exit_price / entry_price - 1) * 100
                row_parts.append(f"off={off:+4d}м:{total:+7.1f}%  ")
            print("".join(row_parts))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ГРАФИКИ
# ═══════════════════════════════════════════════════════════════════════════════

def plot_distribution(ret_dict_sp, ret_dict_im, horizons=HORIZONS, out="1_distribution.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.suptitle("Распределение итоговой доходности ($100/₽100 вложено в начале периода)",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, ret_dict, title in zip(axes, [ret_dict_sp, ret_dict_im], ["S&P 500 (USD)", "IMOEX (RUB)"]):
        data   = [ret_dict[h]["total_ret"].dropna() * 100 for h in horizons]
        labels = [f"{h}л" for h in horizons]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3))
        for patch, color in zip(bp["boxes"], HORIZON_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color="red", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Итоговая доходность, %")
        ax.set_xlabel("Горизонт инвестирования")
        ax.grid(axis="y", alpha=0.35)

    plt.tight_layout()
    add_watermark(fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_decade_table(ret_sp, ret_im, out="2_decades.png"):
    h = 10
    def make_tbl(ret_dict):
        r = ret_dict[h]["cagr"].dropna() * 100
        df = r.to_frame("cagr")
        df["decade"] = (df.index.year // 10) * 10
        tbl = df.groupby("decade")["cagr"].agg(median="median", worst="min", best="max").reset_index()
        tbl.columns = ["Десятилетие", "Медиана CAGR %", "Худший %", "Лучший %"]
        tbl["Десятилетие"] = tbl["Десятилетие"].astype(str) + "-е"
        return tbl

    tbl_sp, tbl_im = make_tbl(ret_sp), make_tbl(ret_im)
    fig, axes = plt.subplots(1, 2, figsize=(15, max(len(tbl_sp), len(tbl_im)) * 0.55 + 2.5))
    fig.suptitle(f"Среднегодовая доходность Buy & Hold — горизонт {h} лет", fontsize=13, fontweight="bold")

    def draw_table(ax, tbl, title):
        ax.axis("off")
        cols = ["Десятилетие", "Медиана CAGR %", "Худший %", "Лучший %"]
        formatted = [[r[0], f"{r[1]:.1f}%", f"{r[2]:.1f}%", f"{r[3]:.1f}%"] for r in tbl[cols].values.tolist()]
        table = ax.table(cellText=formatted, colLabels=cols, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                try:
                    val = float(cell.get_text().get_text().replace("%", ""))
                    if col == 2: cell.set_facecolor("#ffcccc" if val < 0 else "#d5f5d5")
                    elif col == 3: cell.set_facecolor("#d5f5d5")
                    elif col == 1: cell.set_facecolor("#ffcccc" if val < 0 else "#ffffcc" if val < 5 else "#d5f5d5")
                except ValueError: cell.set_facecolor("#f0f0f0")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    draw_table(axes[0], tbl_sp, "S&P 500 (USD)")
    draw_table(axes[1], tbl_im, "IMOEX (RUB)")
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_loss_probability(ret_sp, ret_im, horizons=HORIZONS, out="3_loss_probability.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for ret_dict, label, style in [(ret_sp, "S&P 500", "-o"), (ret_im, "IMOEX", "--s")]:
        probs = [(ret_dict[h]["total_ret"].dropna() < 0).mean() * 100 for h in horizons]
        ax.plot(horizons, probs, style, label=label, linewidth=2, markersize=7)
        for x, y in zip(horizons, probs):
            ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_title("Вероятность убытка P(R < 0) в зависимости от горизонта", fontsize=13, fontweight="bold")
    ax.set_xlabel("Горизонт инвестирования, лет")
    ax.set_ylabel("Вероятность убытка, %")
    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h} лет" for h in horizons])
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.legend()
    ax.grid(alpha=0.35)
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def plot_summary_cagr(ret_sp, ret_im, horizons=HORIZONS, out="4_cagr_summary.png"):
    fig, ax = plt.subplots(figsize=(10, 5))
    x, w = np.arange(len(horizons)), 0.35
    med_sp = [ret_sp[h]["cagr"].dropna().median() * 100 for h in horizons]
    med_im = [ret_im[h]["cagr"].dropna().median() * 100 for h in horizons]
    bars1 = ax.bar(x - w/2, med_sp, w, label="S&P 500", color="#4e79a7")
    bars2 = ax.bar(x + w/2, med_im, w, label="IMOEX",   color="#e15759")
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f"{bar.get_height():.1f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} лет" for h in horizons])
    ax.set_ylabel("Медианная CAGR, %")
    ax.set_title("Медианная среднегодовая доходность (CAGR) по горизонтам", fontsize=13, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.35)
    plt.tight_layout()
    add_watermark(fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


def _make_heatmap_matrix(monthly, max_years=25):
    entry_dates = monthly.index
    data_matrix = []
    for h in range(1, max_years + 1):
        row = []
        months = h * 12
        for i in range(len(entry_dates)):
            target_i = i + months
            if target_i < len(entry_dates):
                row.append(((monthly.iloc[target_i] / monthly.iloc[i]) - 1) * 100)
            else:
                row.append(np.nan)
        data_matrix.append(row)
    return np.array(data_matrix), entry_dates


def _add_crisis_lines(ax, entry_dates, crises, mat_height):
    for i, (label, year, month, color, _desc) in enumerate(crises):
        target = pd.Timestamp(year=year, month=month, day=1)
        if target < entry_dates[0] or target > entry_dates[-1]: continue
        idx = entry_dates.searchsorted(target)
        if idx >= len(entry_dates): continue
        ax.axvline(idx, color="black", linewidth=1.2, alpha=0.7, linestyle="--") # Сделал линии кризисов черными для контраста
        y_pos = mat_height * (0.97 - 0.22 * (i % 4))
        ax.text(idx + 0.5, y_pos, label, color="black", fontsize=7, fontweight="bold",
                rotation=90, va="top", bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.75, ec="none"))


def plot_heatmap(monthly: pd.Series, title: str, out: str,
                 crises=None, vmin_pct=-90, vmax_pct=1000, max_years=25):
    mat, entry_dates = _make_heatmap_matrix(monthly, max_years)

    yearly_mask  = [d.month == 12 for d in entry_dates]
    year_indices = [i for i, m in enumerate(yearly_mask) if m]
    year_labels  = [entry_dates[i].year for i in year_indices]

    step = max(1, len(year_indices) // 35)
    year_indices = year_indices[::step]
    year_labels  = year_labels[::step]

    fig, ax = plt.subplots(figsize=(20, 8))

    # SymLogNorm с настройкой из шапки
    norm = matplotlib.colors.SymLogNorm(
        linthresh=HEATMAP_LINTHRESH, 
        linscale=1.0,
        vmin=vmin_pct,
        vmax=vmax_pct,
        base=10
    )
    
    # Используем палитру из настроек (по умолчанию RdBu_r)
    cmap = plt.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad(color="#E0E0E0") # Серый фон для пустых значений (будущее)

    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm, origin="lower",
                   extent=[0, mat.shape[1], 0.5, max_years + 0.5])

    ax.set_yticks(range(1, max_years + 1))
    ax.set_yticklabels([f"{y}л" for y in range(1, max_years + 1)], fontsize=8)
    ax.set_xticks(year_indices)
    ax.set_xticklabels(year_labels, rotation=90, ha="center", fontsize=8)
    
    ax.set_xlabel("Год входа", fontsize=11)
    ax.set_ylabel("Горизонт инвестирования", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    cbar = plt.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Итоговая доходность, %", fontsize=9)
    
    # Формируем логичные метки для легенды цвета
    cbar_ticks = [-80, -50, -20, 0, 20, 50, 100, 200, 500, 1000, 2000, 3000, 5000]
    cbar_ticks = [t for t in cbar_ticks if vmin_pct <= t <= vmax_pct]
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{t}%" for t in cbar_ticks], fontsize=8)

    if crises:
        _add_crisis_lines(ax, entry_dates, crises, max_years)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color="black", linewidth=1.5, linestyle="--",
                   label=lbl.replace('\n', ' '))
            for lbl, year, _month, _c, _desc in crises
            if entry_dates[0] <= pd.Timestamp(year=year, month=_month, day=1) <= entry_dates[-1]
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9, ncol=2)

    plt.tight_layout()
    add_watermark(fig)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Загружаем данные...")
    sp500 = load_sp500("s-and-p-500.csv")
    imoex = load_imoex("IMOEX.csv")

    print("\nСчитаем доходности...")
    ret_sp = calc_returns(sp500)
    ret_im = calc_returns(imoex)

    print("\n" + "#" * 70)
    print("# ТЕКСТОВЫЙ ЛОГ ДАННЫХ (для анализа LLM)")
    print("#" * 70)
    log_data(sp500, imoex, ret_sp, ret_im)

    print("\n" + "#" * 70)
    print("# ГЕНЕРАЦИЯ ГРАФИКОВ")
    print("#" * 70)

    plot_distribution(ret_sp, ret_im, out="1_distribution.png")
    plot_decade_table(ret_sp, ret_im, out="2_decades.png")
    plot_loss_probability(ret_sp, ret_im, out="3_loss_probability.png")
    plot_summary_cagr(ret_sp, ret_im, out="4_cagr_summary.png")

    plot_heatmap(
        sp500,
        title="Тепловая карта доходности S&P 500  |  X: год входа, Y: горизонт",
        out="5_heatmap_sp500.png",
        crises=SP500_CRISES,
        vmin_pct=-90, 
        vmax_pct=1500,  # <-- Снижен порог vmax для лучшего контраста средних значений
    )

    plot_heatmap(
        imoex,
        title="Тепловая карта доходности IMOEX  |  X: год входа, Y: горизонт",
        out="6_heatmap_imoex.png",
        crises=IMOEX_CRISES,
        vmin_pct=-95, 
        vmax_pct=3000,  # <-- Снижен порог vmax для лучшего контраста
    )

    print("\n✅ Готово.")