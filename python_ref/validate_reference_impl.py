"""
validate_reference_impl.py
Compara reference_impl.py (numpy puro, formulas corregidas) contra golden.npz (librosa).

Es el test de la tesis: si todo cuadra, las formulas son correctas y C++ es traduccion.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from python_ref.reference_impl import run, build_mel_filterbank, hann_periodic, hanning5

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(ROOT, "tests", "golden", "golden.npz")


def rel(a, b):
    denom = np.maximum(np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    g = np.load(GOLDEN)
    y = g["y"].astype(np.float64)
    print("=== reference_impl vs golden (librosa) ===")
    print(f"N samples = {len(y)}")

    R = run(y)

    passed, failed = [], []
    def chk(name, ok, detail=""):
        (passed if ok else failed).append(name)
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name:<28} {detail}")

    # Ventanas (tolerancia a nivel de ULP float64: maxdiff ~ 2e-16 por evaluacion de coseno)
    chk("hann_stft ~bit-exacto (<1e-12)", np.max(np.abs(hann_periodic(2048) - g["hann_stft"])) < 1e-12,
        f"maxdiff={np.max(np.abs(hann_periodic(2048)-g['hann_stft'])):.2e}")
    chk("hann_tg ~bit-exacto (<1e-12)", np.max(np.abs(hann_periodic(344) - g["hann_tg"])) < 1e-12)
    chk("hanning5 bit-exacto", np.array_equal(hanning5(), np.array([0,0.5,1.0,0.5,0])))

    # Mel filterbank (el enorm amplifica el ULP; 2.6e-9 es ruido de maquina, no error de formula)
    mfb = build_mel_filterbank()
    chk("melfb max abs < 1e-8", np.max(np.abs(mfb - g["melfb"])) < 1e-8,
        f"maxdiff={np.max(np.abs(mfb-g['melfb'])):.2e}")

    # STFT power
    chk("S_power MAE rel < 1e-6", mae(R["S_power"], g["S_power"]) < 1e-6,
        f"mae={mae(R['S_power'],g['S_power']):.2e}")

    # Mel spec
    chk("S_mel MAE rel < 1e-5", mae(R["S_mel"], g["S_mel"]) < 1e-5,
        f"mae={mae(R['S_mel'],g['S_mel']):.2e}")

    # dB
    chk("S_dB err abs < 0.01 dB", np.max(np.abs(R["S_dB"] - g["S_dB"])) < 0.01,
        f"maxdiff={np.max(np.abs(R['S_dB']-g['S_dB'])):.2e}")

    # Onset
    c = corr(R["oenv"], g["oenv"])
    chk("oenv corr > 0.9999", c > 0.9999, f"corr={c:.6f} mae={mae(R['oenv'],g['oenv']):.2e}")

    # Tempo
    dt = abs(R["tempo"] - float(g["tempo"]))
    chk("tempo ±0.01 BPM", dt < 0.01, f"ref_impl={R['tempo']:.6f} golden={float(g['tempo']):.6f} dt={dt:.4f}")

    # tg_avg
    ctg = corr(R["tg_avg"], g["tg_avg"])
    chk("tg_avg corr > 0.999", ctg > 0.999, f"corr={ctg:.6f}")

    # localscore
    cl = corr(R["localscore"], g["localscore"])
    chk("localscore corr > 0.999", cl > 0.999, f"corr={cl:.6f}")

    # cumscore
    cc = corr(R["cumscore"], g["cumscore"])
    chk("cumscore corr > 0.999", cc > 0.999, f"corr={cc:.6f}")

    # backlink: % de frames identicos
    same = np.mean(R["backlink"] == g["backlink"])
    chk("backlink 99% identico", same > 0.99, f"{same*100:.2f}% identico")

    # beats
    same_beats = np.array_equal(R["beats"], g["beats_bool"])
    chk("beats bool array identico", same_beats,
        f"mine={int(R['beats'].sum())} golden={int(g['beats_bool'].sum())}")

    # similarity score final (tol 0.04s)
    from python_ref.compare_outputs import similarity_score
    sim = similarity_score(R["beat_times"], g["beat_times"], 0.04)
    chk("similarity >= 93%", sim >= 93.0, f"similarity={sim:.2f}%")

    print()
    print(f"=== RESULTADO: {len(passed)} PASS / {len(failed)} FAIL ===")
    if failed:
        print("FALLOS:", failed)
    return len(failed) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
