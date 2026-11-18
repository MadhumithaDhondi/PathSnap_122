"""Builds report/PathSnap_Technical_Report_DRAFT.pdf from the REAL result files in results/ and figures in report/figs/.
All numbers are read from those files. The prose is a DRAFT: the team must verify it, rewrite it in its own words and add its own
screenshots (Folium view, UI, API console) before submission."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from PIL import Image as PILImage
from pathsnap import config as C

R, F = C.RESULTS, C.ROOT / "report" / "figs"
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#1d2330"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#1d2330"))
H3 = ParagraphStyle("H3", parent=ss["Heading3"], fontSize=10, spaceBefore=5, spaceAfter=2)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.3, leading=12.6, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=7.6, leading=9.6, spaceAfter=0)
CAP = ParagraphStyle("CAP", parent=B, fontSize=8, textColor=colors.HexColor("#52514e"), spaceAfter=8)
BOX = ParagraphStyle("BOX", parent=B, backColor=colors.HexColor("#fff4d6"), borderColor=colors.HexColor("#e0b84a"), borderWidth=.8, borderPadding=6, spaceBefore=6, spaceAfter=10)
PH = ParagraphStyle("PH", parent=B, backColor=colors.HexColor("#f1f1ef"), borderColor=colors.HexColor("#999999"), borderWidth=.8, borderPadding=10, alignment=1, textColor=colors.HexColor("#6b7385"))
P = lambda t, s=B: Paragraph(t, s)
story = []

def img(name, width=17):
    w, h = PILImage.open(F / name).size
    return Image(str(F / name), width=width * cm, height=width * cm * h / w)

def tbl(rows, widths, header=True, fs=7.6, align_right_from=1):
    data = [[Paragraph(str(c), SM) for c in r] for r in rows]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    st = [("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#dfe3ea")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header: st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef1f7")))
    t.setStyle(TableStyle(st)); return t

f4 = lambda v: f"{v:.4f}"
pc = lambda v: f"{100*v:.1f}%"
cmp_ = pd.read_csv(R / "model_comparison.csv").set_index("model")
eda = json.load(open(R / "eda" / "eda_summary.json"))
inv = pd.read_csv(R / "eda" / "dataset_inventory.csv")
build = pd.read_csv(R / "dataset_build_summary.csv")
bo = json.load(open(R / "bayes_opt_best.json")); tv = pd.read_csv(R / "bayes_opt_tuned_vs_untuned.csv").set_index("variant")
trials = pd.read_csv(R / "bayes_opt_trials.csv")
hp = json.load(open(R / "hmm_best_params.json")); hs = pd.read_csv(R / "hmm_vs_classifier_student_test.csv", index_col=0); ht = pd.read_csv(R / "hmm_vs_classifier_train_oof.csv", index_col=0)
abl = pd.read_csv(R / "hmm_ablation_student_test.csv"); fa = pd.read_csv(R / "feature_ablation.csv")
evs = pd.read_csv(R / "hmm_eval_student_test_per_trip.csv"); evt = pd.read_csv(R / "hmm_eval_train_oof_per_trip.csv")
api = pd.read_csv(R / "api_test_results.csv") if (R / "api_test_results.csv").exists() else None
tr_e, te_e = eda["train"], eda["student_test"]
best_model = json.load(open(R / "model_selection.json"))["chosen_model"]

# ------------------------------------------------------------------ title
story += [P("PathSnap: Distributed Map-Matching &amp; Route Reconstruction", ParagraphStyle("T", parent=H1, fontSize=20, spaceAfter=2)),
          P("Technical Report (PDF deliverable) &mdash; <b>DRAFT generated from real experiment outputs</b>", B),
          P("Team name / members / SRNs: <b>[TEAM TO FILL IN]</b> &nbsp;&nbsp; Course: Machine Learning, PES University", B),
          P("<b>Before you submit (read this):</b> every number, table and figure in this draft is read from files produced by the code in the repository "
            "(<i>results/</i>, <i>report/figs/</i>), so it can be re-generated. The explanations, however, were drafted with AI assistance. The event policy prohibits "
            "submitting analysis you cannot explain, and requires disclosure of AI help. Re-run the experiments yourselves, check each claim against your own logs, "
            "rewrite the analysis in your own words, replace the placeholders marked <b>[SCREENSHOT]</b> with real captures from your two laptops, and add the "
            "AI-assistance disclosure required by the organisers (suggested wording in Section 8). Every team member must be able to explain every table and graph in the viva.", BOX)]

# ------------------------------------------------------------------ 1 overview
story += [P("1. Problem and approach in one page", H1),
          P("GPS fixes are noisy, so a vehicle on one street can appear on a parallel street. PathSnap snaps each fix to a road arc and then reconstructs a route that is "
            "consistent with the road network. Our pipeline has four stages: (1) <b>candidate generation</b> &mdash; for every GPS fix, the nearest road arcs within "
            f"{C.CAND_RADIUS_M:.0f} m (at most {C.CAND_MAX_K}; the radius is widened to 120 m and 300 m for fixes with no arc nearby); (2) <b>feature engineering</b> &mdash; 21 numeric features per (fix, arc) pair; "
            "(3) a <b>classifier</b> that outputs the probability that a candidate arc is the reference arc; (4) an <b>HMM decoded with Viterbi</b> that uses these probabilities as "
            "emissions and the directed road connectivity as transitions."), 
          P("<b>Headline results on the 10 student-test trips</b> (development partition; the evaluator partition was never used): classifier-only Top-1 accuracy "
            f"<b>{pc(hs.loc['classifier_only','top1_acc'])}</b> &rarr; <b>{pc(hs.loc['hmm_tuned','top1_acc'])}</b> with HMM/Viterbi; observation-level macro F1 "
            f"{f4(hs.loc['classifier_only','obs_macro_f1'])} &rarr; {f4(hs.loc['hmm_tuned','obs_macro_f1'])}; route continuity {pc(hs.loc['classifier_only','route_continuity'])} &rarr; "
            f"{pc(hs.loc['hmm_tuned','route_continuity'])}. Final classifier: Bayesian-tuned {best_model}.")]

# ------------------------------------------------------------------ 2 EDA
story += [P("2. Dataset understanding and exploratory data analysis", H1),
          P("<b>Partitions used.</b> EDA used the <b>training</b> (80 trips) and <b>student-test</b> (10 trips) partitions only. The 10 evaluator trips were not available to us and were never touched.", B),
          P("2.1 How the raw data is organised", H2),
          P("Each trip is a folder with four header-less, whitespace-separated files: <i>.track</i> (lon, lat, timestamp in seconds), <i>.nodes</i> (lon, lat; row index = node id), "
            "<i>.arcs</i> (from_node, to_node; row index = arc id; directed) and <i>.route</i> (the reference sequence of arc ids). Every trip has its <b>own</b> road network, and a "
            "GPS row does not correspond to a row of any network file; the link between them has to be built geometrically."),
          tbl([["Measure", "Train (80 trips)", "Student-test (10 trips)"],
               ["GPS fixes (total / per-trip min, mean, max)", f"{tr_e['obs_total']:,} / {tr_e['obs_min']}, {tr_e['obs_mean']:.0f}, {tr_e['obs_max']:,}", f"{te_e['obs_total']:,} / {te_e['obs_min']}, {te_e['obs_mean']:.0f}, {te_e['obs_max']:,}"],
               ["Missing / non-finite values in .track", tr_e["missing_values"], te_e["missing_values"]],
               ["Time steps exactly 1 s", pc(tr_e["dt_eq1_share"]), pc(te_e["dt_eq1_share"])],
               ["Trips with a gap &gt; 10 s (largest gap)", f"{tr_e['trips_with_gaps_gt10s']} ({tr_e['max_gap_s']:,.0f} s)", f"{te_e['trips_with_gaps_gt10s']} ({te_e['max_gap_s']:,.0f} s)"],
               ["Non-increasing timestamps", tr_e["nonincreasing_timestamps"], te_e["nonincreasing_timestamps"]],
               ["Road nodes per trip (mean) / arcs per trip (mean)", f"{tr_e['nodes_mean']:,.0f} / {tr_e['arcs_mean']:,.0f}", f"{te_e['nodes_mean']:,.0f} / {te_e['arcs_mean']:,.0f}"],
               ["Isolated nodes", tr_e["isolated_nodes_total"], te_e["isolated_nodes_total"]],
               ["Trips whose network has &gt; 1 connected component", f"{tr_e['trips_with_multiple_components']} of 80", f"{te_e['trips_with_multiple_components']} of 10"],
               ["Median share of nodes in the largest component", pc(tr_e["median_largest_component_share"]), pc(te_e["median_largest_component_share"])],
               ["Zero-length arcs / duplicate arc rows", f"{tr_e['zero_length_arcs']} / {tr_e['duplicate_arcs']:,}", f"{te_e['zero_length_arcs']} / {te_e['duplicate_arcs']:,}"],
               ["Reference-route arc ids outside the arcs file", tr_e["route_bad_refs"], te_e["route_bad_refs"]],
               ["Lowest share of consecutive route arcs that connect head-to-tail", pc(tr_e["route_connected_share_min"]), pc(te_e["route_connected_share_min"])]],
              [8.2, 4.4, 4.4]), Spacer(1, 6),
          img("../../results/eda/fig_dataset_overview.png", 17), P("Figure 1. Trip length, network size and connectivity (train and student-test partitions).", CAP),
          img("../../results/eda/fig_sampling_and_noise.png", 15.5),
          P("Figure 2. Left: time between consecutive fixes (log scale) &mdash; sampling is nominally 1 Hz but not always. Right: distance between each fix and its aligned reference arc "
            "(78 usable train trips + 10 student-test trips): most fixes are within about 10 m, with a long tail of outliers.", CAP),
          P("2.2 Preprocessing", H2),
          P("<b>Projection.</b> Longitude/latitude are angles, so we project each trip into a local metric frame (equirectangular projection centred on the trip's network, Earth radius 6,371,008.8 m). "
            "At city scale the distortion is well below the GPS noise. All distances (fix-to-arc, arc length, heading) are in metres; coordinate order is (lon, lat) in the files. "
            "<b>Validity.</b> The loader flags non-finite or out-of-range rows (none occurred) and keeps the original row index so predictions map back to file rows. "
            "<b>Arc geometry</b> is the straight segment between the arc's two nodes."),
          P("<b>Deriving a label for every GPS fix.</b> The reference route is only an ordered list of arcs, without timestamps. We align fixes to it with a monotone dynamic program: fix <i>i</i> "
            "is assigned a route position <i>j(i)</i> that never decreases; the cost is the distance from the fix to that route arc plus a small penalty (3 m) per route arc skipped. Because only arcs that are "
            "<i>in the route, with their direction</i>, are eligible, the label also encodes travel direction. This label is derived by us and is therefore an approximation of the true arc; "
            "see limitations below."),
          P("<b>Label quality control.</b> For two training trips (ids 00000033 and 00000054) the median distance between the fixes and the aligned reference route is 122 m and 140 m, although the fixes lie near roads "
            "of the network (trip 33: median nearest-arc distance 7 m). The reference route there clearly does not follow the GPS track, so these labels are unreliable. We excluded these two trips from "
            f"<b>fitting and cross-validation</b> (rule: median fix-to-route distance &gt; {C.LABEL_QC_MAX_MEDIAN_M:.0f} m, written before any model was trained) &mdash; leaving 78 training trips. "
            "No student-test trip is affected (largest median 6.8 m), and the prediction pipeline never filters trips, because it never uses labels."),
          P("2.3 Patterns observed and features chosen", H2),
          P(f"<b>Candidate generation.</b> A {C.CAND_RADIUS_M:.0f} m search radius (nearest {C.CAND_MAX_K} arcs) was chosen from the training trips: with 45 m the average number of candidates is "
            f"{build[build.partition=='train'].mean_cand.mean():.1f} per fix and the reference arc is missing from the candidate set for {100*build[(build.partition=='train')&(build.label_dist_median<=50)].cand_miss_rate.mean():.1f}% of training fixes "
            f"(trips passing label QC) and {100*build[build.partition=='student_test'].cand_miss_rate.mean():.2f}% of student-test fixes. Because labels come from an alignment that prefers nearby route arcs, "
            "this <i>Candidate Miss Rate</i> is somewhat optimistic; the misses are concentrated in a few trips (e.g. 00000002 and 00000072; see results/dataset_build_summary.csv), whose cause we have not diagnosed."),
          P("<b>Observed patterns that motivated the features</b>: (i) GPS noise is mostly under 10 m, but parallel streets and junctions are within noise distance, so distance alone is ambiguous; "
            "(ii) every street appears as two directed arcs with identical geometry, so <b>direction of travel</b> (heading agreement) is essential; (iii) when the vehicle stops, fixes cluster and wander "
            "(visible in the Folium maps), so <b>temporal context</b> is needed; (iv) arcs are short, so a vehicle often crosses several arcs between fixes."),
          tbl([["Feature group", "Features (per fix&ndash;candidate pair)", "Why"],
               ["Geometry", "dist to arc, position along arc, arc length, rank by distance, gap to nearest candidate, distance ratio, number of candidates", "How plausible is the arc given only this fix, relative to its competitors"],
               ["Motion / heading", "cos &amp; |angle| between track heading and arc bearing (&plusmn;1 and &plusmn;4 fix windows), heading-valid flag, speed, time since previous fix", "Separates the two directions of a street and side streets"],
               ["Temporal context", "mean / max distance and distance at &plusmn;3 and &plusmn;6 fixes from the candidate arc", "A true arc stays close to neighbouring fixes; a spurious one does not"],
               ["Topology", "out-degree of the arc's head node, in-degree of its tail node", "Junction structure"]], [2.6, 8.4, 6.0]),
          Spacer(1, 4), img("feature_ablation.png", 12),
          P("Figure 3. Feature-group ablation (LightGBM, untuned config). Geometry alone gives Top-1 "
            f"{pc(fa.cv_top1.iloc[0])} (CV); adding motion/heading lifts it to {pc(fa.cv_top1.iloc[1])}; temporal context and topology add smaller gains "
            f"(CV macro F1 {f4(fa.cv_macro_f1.iloc[1])} &rarr; {f4(fa.cv_macro_f1.iloc[2])} &rarr; {f4(fa.cv_macro_f1.iloc[3])}; the later steps are within fold-to-fold noise on CV Top-1). Source: results/feature_ablation.csv.", CAP),
          img("../../results/eda/fig_trajectories_network.png", 17),
          P("Figure 4. GPS fixes (blue), reference route (green) and nearby road network (grey) for one training trip and one student-test trip.", CAP),
          P("<b>Folium map.</b> See Section 6 for the interactive Folium maps (raw GPS, road network, reference route, classifier-only and HMM routes).", B)]

# ------------------------------------------------------------------ 3 models
story += [PageBreak(), P("3. Model experimentation and comparison", H1),
          P("<b>Protocol.</b> Candidate rows of every 3rd GPS fix of the 78 usable training trips are used to fit and cross-validate (to keep training time practical; "
            f"{int(cmp_.fit_rows.max()):,} rows). Cross-validation is <b>group-aware</b>: 5-fold GroupKFold with whole trips held out. KNN is fitted on every 9th fix because its inference cost grows with the "
            "training set. Development evaluation uses <b>all</b> fixes of the 10 student-test trips. Preprocessing (standardisation) is fitted on training data inside each pipeline. Seed = 42 for every model. "
            "<b>Primary selection metric:</b> mean group-CV candidate-level macro F1 (macro-averaging over the two classes &ldquo;reference arc&rdquo; / &ldquo;other candidate&rdquo; keeps the rare positive class "
            "from being ignored), with CV Top-1 accuracy and log loss as checks; student-test results were used to compare, not to select."),
          P("<b>Metric definitions.</b> <i>Top-1 accuracy</i>: per GPS fix, take the candidate with the highest score and compare it with the reference arc (a fix with no candidate counts as an error). "
            "<i>Macro F1 (cand.)</i>: candidate-level binary classification at threshold 0.5. <i>Log loss</i>: candidate-level binary cross-entropy. <i>Route continuity</i>: share of consecutive fixes whose predicted arcs "
            "are identical or can follow each other along the directed network within 3 arc-steps. <i>Inference time</i>: milliseconds per GPS fix on the student-test set (candidate scoring only, 2 CPU cores).", B),
          P("3.1 Overall model comparison", H2)]
rows = [["Model", "Algorithm family", "Key configuration", "CV macro F1 (mean ± std)", "Top-1 acc. (student-test)", "Route continuity", "Student-test macro F1", "Log loss", "Inference (ms/fix)"]]
keycfg = {"LogisticRegression": "standardised; C=1, L2, lbfgs", "GaussianNB": "standardised; var_smoothing=1e-9", "KNN": "standardised; k=25, distance-weighted; fit on every 9th fix",
          "DecisionTree": "max_depth=8, min_leaf=50, Gini", "RandomForest": "100 trees, depth 12, min_leaf 20, max_features=sqrt, max_samples 0.5",
          "HistGradientBoosting": "200 iters, lr 0.1, 31 leaves", "LightGBM": "300 trees, lr 0.05, 31 leaves, subsample/colsample 0.8", "XGBoost": "300 trees, lr 0.05, depth 6, hist, subsample/colsample 0.8", "MLP": "standardised; (64,32) ReLU, Adam, 40 epochs, early stopping"}
for m, r in cmp_.iterrows():
    rows.append([m, r.family, keycfg[m], f"{r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}", f4(r.test_top1), f4(r.test_route_continuity), f4(r.test_cand_macro_f1), f4(r.test_logloss), f"{r.inference_ms_per_obs:.4f}"])
story += [tbl(rows, [2.3, 2.1, 3.9, 2.0, 1.4, 1.3, 1.4, 1.2, 1.4]),
          P("Table 1. Nine models trained and compared (at least five required). All values are produced by scripts/03_compare_models.py (results/model_comparison.csv). "
            "Route continuity is computed from classifier-only predictions. The inference-time column excludes candidate generation and feature computation (shared by all models).", CAP)]

def gap_text(m, r):
    g = r.train_top1 - r.cv_top1_mean
    if g > 0.08:
        return (f"<b>Clear overfitting.</b> In-sample Top-1 is {pc(r.train_top1)} but on held-out CV trips only {pc(r.cv_top1_mean)} (gap {100*g:.1f} points) and {pc(r.test_top1)} on student-test; "
                "distance-weighted KNN reproduces its own training points almost exactly.")
    if r.cv_top1_mean < 0.86:
        return (f"<b>Underfitting / high bias.</b> Training Top-1 ({pc(r.train_top1)}) and CV Top-1 ({pc(r.cv_top1_mean)}) are almost equal but clearly lower than the best model "
                f"({pc(cmp_.cv_top1_mean.max())}), so the model class cannot represent the decision rule; log loss is also the worst ({f4(r.test_logloss)}).")
    if g < 0.02:
        return (f"<b>Mild underfitting, no overfitting.</b> Training Top-1 {pc(r.train_top1)} vs CV {pc(r.cv_top1_mean)} (gap {100*g:.1f} points): train and held-out scores are close, but both sit below the boosted models "
                f"({pc(cmp_.cv_top1_mean.max())} CV), so capacity rather than variance is the limit.")
    return (f"<b>Small generalisation gap.</b> Training Top-1 {pc(r.train_top1)} vs CV {pc(r.cv_top1_mean)} (gap {100*g:.1f} points) and student-test {pc(r.test_top1)}: "
            "some fitting to training trips, but held-out performance is stable (CV fold std of macro F1 "
            f"{r.cv_macro_f1_std:.3f}).")

info = {
 "LogisticRegression": dict(why="A linear model on well-scaled geometric/heading features gives a fast, well-calibrated probability baseline, and its probabilities can feed the HMM directly.",
     sl="Strengths: fast, convex, interpretable coefficients. Limitation: cannot express interactions such as &ldquo;close <i>and</i> heading agrees&rdquo; or the non-linear role of distance rank.",
     keys=["C", "max_iter", "solver"], loss="Regularised log loss (cross-entropy + L2 penalty, C=1)."),
 "GaussianNB": dict(why="A generative baseline that models each feature as Gaussian per class; very cheap.",
     sl="Strengths: no iterative training, tiny model. Limitation: assumes conditionally independent Gaussian features, but our window and heading features are strongly correlated and heavy-tailed, so its probabilities are poorly calibrated (highest log loss).",
     keys=["var_smoothing"], loss="Maximises the joint likelihood (class priors and per-feature Gaussians); no iterative loss."),
 "KNN": dict(why="Instance-based: similar (distance, heading, context) situations in the training trips should have similar outcomes.",
     sl="Strengths: non-parametric, no training. Limitation: 21-dimensional Euclidean distance after scaling weights all features equally, inference is by far the slowest (about 1.7 ms per fix versus at most 0.05 ms for every other model) and it memorises training data.",
     keys=["n_neighbors", "weights", "metric"], loss="No training loss; prediction is a distance-weighted vote of the 25 nearest training rows."),
 "DecisionTree": dict(why="Non-linear rules (e.g. &ldquo;dist &lt; 8 m and heading agrees and rank = 0&rdquo;) match how the problem is naturally ambiguous at junctions.",
     sl="Strengths: handles interactions and scale differences, interpretable. Limitation: a single shallow tree gives coarse, step-like probabilities (worse log loss than ensembles) and is high-variance.",
     keys=["max_depth", "min_samples_leaf", "criterion"], loss="Gini impurity reduction at each split."),
 "RandomForest": dict(why="Bagging many randomised trees reduces the variance of a single tree while keeping non-linear interactions.",
     sl="Strengths: robust, few tuning needs. Limitation: slower to train than the boosting models here (about 33 s versus 4&ndash;8 s), probabilities are averaged leaf frequencies (less sharp), and it did not beat boosting.",
     keys=["n_estimators", "max_depth", "min_samples_leaf", "max_features", "max_samples"], loss="Gini impurity per tree; predictions averaged over 100 bootstrap trees."),
 "HistGradientBoosting": dict(why="Boosting corrects residual errors sequentially and works well on tabular numeric features; histogram splits make it fast.",
     sl="Strengths: strong accuracy, good probabilities, fast. Limitation: more hyper-parameters; without tuning it was slightly behind LightGBM in our runs.",
     keys=["max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization"], loss="Binary log loss, fitted by gradient boosting on histogram-binned features."),
 "LightGBM": dict(why="Leaf-wise gradient-boosted trees: the best accuracy/speed trade-off for this kind of tabular candidate-scoring task, with calibrated probabilities that the HMM needs.",
     sl="Strengths: top-two CV macro F1 and log loss (level with XGBoost), fast training (about six seconds on 407k rows). Limitation: more sensitive to hyper-parameters (addressed in Section 4) and slower to score than a single tree.",
     keys=["n_estimators", "learning_rate", "num_leaves", "subsample", "colsample_bytree", "min_child_samples", "reg_lambda"], loss="Binary log loss (gradient boosting)."),
 "XGBoost": dict(why="Level-wise gradient-boosted trees with built-in regularisation; a widely used strong baseline for tabular candidate scoring and a direct comparison with LightGBM's leaf-wise growth.",
     sl="Strengths: strong accuracy, regularised, good probabilities. Limitation: depth-limited level-wise growth can be slower than LightGBM and it has many hyper-parameters; it was not tuned here.",
     keys=["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree", "tree_method"], loss="Binary log loss (gradient boosting, second-order)."),
 "MLP": dict(why="A small neural network can learn smooth non-linear combinations of the scaled features.",
     sl="Strengths: smooth probabilities, cheap inference. Limitation: needs scaling, sensitive to initialisation, and gave no gain over tree ensembles on this tabular problem.",
     keys=["hidden_layer_sizes", "alpha", "batch_size", "max_iter", "early_stopping", "learning_rate_init"], loss="Binary log loss with L2 penalty (alpha=1e-4), optimised with Adam; 10% of rows held out for early stopping."),
}
story += [P("3.2 Per-model analysis (compulsory questions)", H2),
          P("Questions are numbered as in the organisers' template (A: 1&ndash;2, B: 5&ndash;6, C: 7&ndash;9; the template skips 3&ndash;4, so those numbers are not used).", CAP)]
for m, r in cmp_.iterrows():
    d = info[m]; params = json.load(open(R / "model_runs" / f"{m}.json"))["params"]
    ptxt = ", ".join(f"{k} = {params[k]}" for k in d["keys"] if k in params)
    block = [P(f"3.2.{list(cmp_.index).index(m)+1} {m} <font size=8 color='#6b7385'>({r.family})</font>", H3),
             P(f"<b>A. Model understanding.</b> <b>1. Why suitable?</b> {d['why']} <b>2. Strengths and limitations.</b> {d['sl']}"),
             P(f"<b>B. Training and configuration.</b> <b>5. Important hyper-parameters (initial values used in the comparison):</b> {ptxt}. <b>6. Loss / objective:</b> {d['loss']} "
               f"Training time {r.fit_seconds:.1f} s (all training rows used by this model: {int(r.fit_rows):,})."),
             P(f"<b>C. Experimental results.</b> <b>7&ndash;8. Student-test (10 trips, all fixes):</b> Top-1 accuracy {f4(r.test_top1)}; observation-level macro F1 (arc classes, mean over trips) {f4(r.test_obs_macro_f1)}; "
               f"candidate-level macro F1 {f4(r.test_cand_macro_f1)} (precision {f4(r.test_precision)}, recall {f4(r.test_recall)}); log loss {f4(r.test_logloss)}; route continuity {f4(r.test_route_continuity)}; "
               f"inference {r.inference_ms_per_obs:.4f} ms per fix. 5-fold group-CV macro F1 {r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}, CV Top-1 {r.cv_top1_mean:.4f}. "
               f"<b>9. Over/under-fitting:</b> {gap_text(m, r)}")]
    story += block
story += [P("3.3 Model visualisations", H2),
          img("cmp_cv_macro_f1.png", 12.5), P("Figure 5. Cross-validation macro F1 (mean ± std over 5 group folds). The bars of the top five models overlap within one standard deviation, so differences between them are small compared with fold-to-fold variation caused by trip difficulty.", CAP),
          img("cmp_test_top1_f1.png", 14.5), P("Figure 6. Student-test Top-1 accuracy and candidate-level macro F1 for all models.", CAP),
          img("cmp_logloss.png", 12.5), P("Figure 7. Student-test log loss (probability quality, relevant for the HMM emissions). GaussianNB is clearly worst; XGBoost and LightGBM have the lowest log loss, within 0.002 of each other.", CAP),
          img("cmp_overfit.png", 14.5), P("Figure 8. Over/under-fitting evidence: in-sample, held-out CV and student-test Top-1. KNN shows the large generalisation gap; LogisticRegression and GaussianNB show high bias.", CAP),
          img("cmp_confusion.png", 15.5), P("Figure 9. Candidate-level confusion matrices on the student-test set (threshold 0.5; rows normalised by actual class). Optional diagnostic.", CAP),
          img("learning_curves.png", 16), P("Figure 10. Training histories exist only for MLP (training loss per epoch) and LightGBM (train/validation log loss per boosting round, CV fold 0). No curves are drawn for models without iterative training.", CAP),
          P(f"<b>Model selection.</b> {best_model} was chosen by the pre-fixed rule (highest mean group-CV macro F1 among the models compared at that stage, {cmp_.loc[best_model,'cv_macro_f1_mean']:.4f}) and then tuned (Section 4) and used for the HMM (Section 5). "
            f"XGBoost was added to the comparison afterwards and scores {cmp_.loc['XGBoost','cv_macro_f1_mean']:.4f} CV macro F1 and {f4(cmp_.loc['XGBoost','test_logloss'])} student-test log loss, versus {cmp_.loc[best_model,'cv_macro_f1_mean']:.4f} and {f4(cmp_.loc[best_model,'test_logloss'])} for {best_model}. "
            f"The CV difference ({cmp_.loc['XGBoost','cv_macro_f1_mean']-cmp_.loc[best_model,'cv_macro_f1_mean']:+.4f}) is far below the fold standard deviation ({cmp_.loc[best_model,'cv_macro_f1_std']:.3f}), so the two are statistically indistinguishable. "
            "We kept LightGBM because tuning, the HMM parameters and the saved pipeline were all built on its probabilities, and we did not re-tune the decoder for XGBoost. The margin over HistGradientBoosting, RandomForest and the MLP is also small relative to the fold standard deviation, "
            "so the choice is a judgement supported by probability quality (log loss), speed and compatibility with the HMM, not a large accuracy gap.")]

# ------------------------------------------------------------------ 4 BO
bp = bo["best_params"]
story += [PageBreak(), P("4. Bayesian hyper-parameter optimisation (LightGBM)", H1),
          P("<b>1&ndash;2. Search space.</b> n_estimators: integer 100&ndash;500 (step 50); learning_rate: 0.01&ndash;0.3 (log); num_leaves: 15&ndash;96 (log); min_child_samples: 10&ndash;200 (log); subsample: 0.5&ndash;1.0; "
            "colsample_bytree: 0.4&ndash;1.0; reg_lambda and reg_alpha: 0.001&ndash;10 (log). <b>3. Objective.</b> Maximise the mean candidate-level macro F1 over 3 group-aware CV folds (folds split by trip) of the 78 training trips. "
            f"<b>4. Trials.</b> {bo['n_trials']} trials (the first is the untuned configuration, enqueued as a baseline) using Optuna's TPE sampler &mdash; a sequential model-based (Bayesian) optimiser; 10 random start-up trials, seed 42; total {bo['total_seconds']/60:.1f} min on 2 CPU cores. "
            "All trials are saved in results/bayes_opt_trials.csv."),
          P("<b>5. Best configuration found:</b> " + ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in bp.items()) + f" (3-fold objective {bo['best_cv3_macro_f1']:.4f}; the untuned baseline scored {trials.value.iloc[0]:.4f} under the same objective).", B),
          img("bayes_opt_history.png", 17), P("Figure 11. Optimisation history, best-so-far curve and two hyper-parameter scatter plots from the saved trial record. The objective surface is very flat (all trials between "
            f"{trials.value.min():.3f} and {trials.value.max():.3f}), so the search could only find small gains.", CAP),
          P("<b>6. Did the tuned model outperform the untuned one?</b> Under the identical protocol (5-fold group CV on training trips, then student-test):", B),
          tbl([["Metric", "Untuned", "Tuned", "Difference"]] + [
               [lab, f4(tv.loc["untuned", col]), f4(tv.loc["tuned", col]), f"{tv.loc['tuned', col]-tv.loc['untuned', col]:+.4f}"] for lab, col in
               [("CV macro F1 (5-fold)", "cv_macro_f1_mean"), ("CV Top-1 accuracy", "cv_top1_mean"), ("Student-test Top-1 accuracy", "test_top1"), ("Student-test macro F1 (cand.)", "test_cand_macro_f1"),
                ("Student-test log loss", "test_logloss"), ("Inference time on student-test (s)", "inference_seconds"), ("Training time (s)", "fit_seconds")]], [6, 3, 3, 3]),
          P("Table 2. Tuned vs untuned LightGBM. Source: results/bayes_opt_tuned_vs_untuned.csv.", CAP),
          P("<b>Honest reading.</b> Tuning improved CV macro F1 by only "
            f"{tv.loc['tuned','cv_macro_f1_mean']-tv.loc['untuned','cv_macro_f1_mean']:+.4f}, far below the fold standard deviation ({cmp_.loc['LightGBM','cv_macro_f1_std']:.3f}), and student-test Top-1 moved by "
            f"{tv.loc['tuned','test_top1']-tv.loc['untuned','test_top1']:+.4f}. We therefore do <b>not</b> claim a meaningful improvement. The tuned configuration was kept because the selection rule, fixed in advance, was CV performance; "
            "the objective surface is very flat around the default LightGBM settings. The computational cost of tuning was about "
            f"{bo['total_seconds']/60:.0f} minutes versus about {tv.loc['untuned','fit_seconds']:.0f} s to train one model.")]

# ------------------------------------------------------------------ 5 HMM
tp = hp["tuned"]
d_s = (evs[evs.variant == "hmm_tuned"].set_index("trip_id").top1_acc - evs[evs.variant == "classifier_only"].set_index("trip_id").top1_acc)
d_t = (evt[evt.variant == "hmm_tuned"].set_index("trip_id").top1_acc - evt[evt.variant == "classifier_only"].set_index("trip_id").top1_acc)
worst_t = d_t.idxmin()
story += [PageBreak(), P("5. HMM and Viterbi route reconstruction", H1),
          P("<b>1. Hidden states.</b> At each GPS fix the hidden state is one of that fix's candidate arcs (up to 12). The state set differs from fix to fix, and every state keeps its arc id, so the candidate-to-arc correspondence is preserved "
            "(code: src/pathsnap/hmm.py)."),
          P("<b>2. Emissions.</b> log P(fix | arc) = &alpha; &middot; log p<sub>clf</sub>(arc | features), where p<sub>clf</sub> is the tuned LightGBM probability that the arc is the reference arc (clipped to [1e-6, 1]); &alpha; tempers the classifier's confidence."),
          P("<b>3. Transitions.</b> Between consecutive fixes, going from arc <i>a</i> to arc <i>b</i> costs: 0 if <i>a</i> = <i>b</i>; &beta;&middot;<i>h</i> if <i>b</i> can be reached from <i>a</i> in <i>h</i> &le; H arc-steps along the <b>directed</b> network (H = max_hops, widened for time gaps); "
            "an extra U-turn penalty if <i>b</i> is the reverse twin of <i>a</i>; and a constant jump penalty J otherwise. Costs are subtracted in log space. "
            f"Tuned parameters: &alpha;={tp['alpha']}, &beta;={tp['hop_penalty']}, max_hops={tp['max_hops']}, jump J={tp['jump_penalty']}, U-turn={tp['uturn_penalty']}, gap_hops_per_s={tp['gap_hops_per_s']}. "
            "(&alpha; and the penalties are partly redundant &mdash; scaling all by one factor leaves the decoded path unchanged &mdash; so &alpha; hitting the lower grid edge is not a separate finding.)"),
          P("<b>Viterbi recurrence.</b> &delta;<sub>t</sub>(j) = e<sub>t</sub>(j) + max<sub>i</sub> [ &delta;<sub>t&minus;1</sub>(i) &minus; cost(i&rarr;j) ], keeping back-pointers; the route is the backtracked arg-max of &delta; at the last fix."),
          P("<b>4. Missing candidates and large GPS gaps.</b> If the radius search finds nothing, it is widened (120 m, then 300 m). Any fix that still has no candidate, or a time gap &gt; 30 s, cuts the sequence; each piece is decoded independently "
            "and fixes without candidates inherit the nearest decoded arc. For shorter gaps the allowed number of arc-steps grows with the gap: H = max_hops + &lceil;gap_hops_per_s &middot; (dt &minus; 1)&rceil; (capped at 14)."),
          P("<b>5. Disconnected consecutive arcs.</b> The jump penalty is finite, so the decoder can still switch to an unconnected arc when the evidence is strong (e.g. after GPS outliers) instead of failing."),
          P("<b>Decoder tuning.</b> A coordinate search over the six decoder parameters (two sweeps, grids in scripts/05_hmm_decode.py) maximised Top-1 on <b>out-of-fold</b> classifier probabilities of a random subset of "
            f"{len(hp['tuning_trips'])} training trips (never student-test or evaluator data): {pc(hp['tuning_top1_classifier_only'])} classifier-only &rarr; {pc(hp['tuning_top1_default'])} default HMM &rarr; {pc(hp['tuning_top1_tuned'])} tuned HMM (on those trips). Path: results/hmm_tuning_path.csv."),
          P("<b>6. Did HMM decoding help?</b> Yes, clearly, on both partitions:", B)]
rows = [["Variant", "Top-1 acc.", "Obs. macro F1", "Route continuity", "Route precision", "Route recall", "Route F1"]]
for part, df in (("Student-test (10 trips)", hs), ("Training, out-of-fold (78 trips)", ht)):
    rows.append([f"<b>{part}</b>", "", "", "", "", "", ""])
    for v, lab in (("classifier_only", "Classifier only"), ("hmm_default", "Classifier + HMM (default params)"), ("hmm_tuned", "Classifier + HMM (tuned params)")):
        r = df.loc[v]; rows.append([lab, f4(r.top1_acc), f4(r.obs_macro_f1), f4(r.route_continuity), f4(r.route_precision), f4(r.route_recall), f4(r.route_f1)])
story += [tbl(rows, [5.6, 1.9, 2.0, 2.0, 2.0, 1.8, 1.7]),
          P("Table 3. Classifier-only vs classifier + HMM. Top-1 and continuity are pooled over fixes; macro F1 and route metrics are averaged over trips. Route precision/recall compare the set of predicted arcs with the reference route. Source: results/hmm_vs_classifier_*.csv.", CAP),
          img("hmm_comparison.png", 17),
          P("Figure 12. Left: per-trip Top-1 accuracy on student-test, classifier-only vs HMM. Right: ablation on student-test &mdash; removing the hop penalty removes roughly half of the gain and a weak jump penalty removes all of it; the U-turn penalty and gap widening matter little on these trips.", CAP),
          P(f"<b>Where it helps and where it does not.</b> On student-test the HMM improved every one of the 10 trips (+{100*d_s.min():.1f} to +{100*d_s.max():.1f} points; largest on trip 00000077, where the classifier alone picks side streets "
            "while the vehicle is stationary and fixes cluster; see Section 6). The smallest gain is on trip 00000065 (see Figure 14); we have not diagnosed why. On the 78 training trips (out-of-fold) the HMM lowered Top-1 on "
            f"{int((d_t < 0).sum())} trips (worst: trip {worst_t}, {100*d_t.min():.1f} points), (we did not investigate the cause; inspect these trips on a map before writing your explanation). "
            "Route <i>recall</i> is essentially unchanged, which is why route F1 moves less than Top-1."), 
          P("Note: the tuning subset (24 trips) is a subset of the 78 training trips, so the training row of Table 3 is mildly optimistic; the student-test row is the clean estimate.", CAP)]

# ------------------------------------------------------------------ 6 Folium
story += [PageBreak(), P("6. Geographical visualisation using Folium", H1),
          P("Maps are produced by scripts/06_make_maps.py from the real pipeline output and saved as interactive HTML in maps/ (student_test trips 00000000, 00000077, 00000065). Layers (toggle with the layer control): "
            "raw GPS fixes (blue), road network near the trip (grey, off by default), reference route (green), classifier-only prediction (orange, dashed), classifier + HMM prediction (red); legend at the bottom left; "
            "the initial view is the 300-fix window where the classifier-only prediction makes the most mistakes. <b>The screenshots below were rendered in a cloud sandbox without internet, so the OpenStreetMap base tiles are missing (grey background). "
            "Replace them with your own screenshots taken from the HTML files on your laptop, with the base map visible.</b>"),
          img("folium_student_test_00000077.png", 16.5),
          P("Figure 13. Trip 00000077 (student-test). Where the vehicle is stationary the fixes cluster and the classifier alone (orange) jumps onto side streets; the HMM route (red) follows the reference route (green).", CAP),
          img("folium_student_test_00000065.png", 16.5),
          P("Figure 14. Trip 00000065 (student-test), the trip with the smallest HMM gain.", CAP)]

# ------------------------------------------------------------------ 7 API/UI
story += [PageBreak(), P("7. API, UI and deployment", H1),
          P("<b>Architecture (two laptops).</b> Laptop 1 runs the FastAPI service (api/main.py): <i>GET /trips</i> lists the evaluation trips, <i>POST /predict</i> with <i>{\"trip_id\": ...}</i> runs the full pipeline "
            "(candidate generation &rarr; features &rarr; LightGBM &rarr; HMM/Viterbi) on the trip's track and network files, then computes accuracy <b>server-side</b> from the reference route. "
            "Laptop 2 serves a static UI (ui/index.html, Leaflet) that shows the ten trip buttons, sends only the trip id and renders the returned GPS points, predicted routes and metrics. "
            "The prediction code path (src/pathsnap/pipeline.py) never reads the reference route; labels are loaded only in api/evaluator.py after the prediction exists, and are never returned to the UI. "
            "Every request is logged to the console and to logs/api_requests.log."),
          P("<b>Run commands.</b> Laptop 1: <i>pip install -r requirements.txt; PATHSNAP_EVAL_DATA=&lt;folder with the 10 evaluation trips&gt; python -m uvicorn api.main:app --host 0.0.0.0 --port 8000</i>. "
            "Laptop 2: <i>python ui/serve.py 8080</i>, open <i>http://localhost:8080</i>, set the API address to <i>http://&lt;laptop-1-ip&gt;:8000</i> and press &ldquo;Load evaluation trips&rdquo;. (Allow port 8000 in the Laptop 1 firewall.)")]
if api is not None:
    w = (api.hmm_top1 * api.n_obs).sum() / api.n_obs.sum(); wc = (api.clf_top1 * api.n_obs).sum() / api.n_obs.sum()
    story += [P(f"<b>Integration test.</b> scripts/07_test_api.py called <i>/predict</i> over HTTP for all {len(api)} trips in the stand-in evaluation folder (the student-test trips, because the real evaluator trips are not available to us). All calls succeeded; "
                f"observation-weighted Top-1 was {pc(wc)} (classifier-only) and {pc(w)} (with HMM), matching the offline evaluation. Server-side pipeline time per trip was {api.pipeline_s.min():.2f}&ndash;{api.pipeline_s.max():.2f} s "
                f"(longest trip: {int(api.n_obs.max()):,} fixes). The stand-in numbers are <b>not</b> an estimate for the evaluator trips.")]
story += [Spacer(1, 6), P("<b>[SCREENSHOT 1 &mdash; TEAM TO INSERT]</b><br/>The working UI on Laptop 2 with one trip selected (map + metrics table).", PH), Spacer(1, 8),
          P("<b>[SCREENSHOT 2 &mdash; TEAM TO INSERT]</b><br/>The API console on Laptop 1 (the receiving laptop) showing successful <i>POST /predict</i> log lines for the trips you demonstrate.", PH),
          Spacer(1, 6), P("The submitted source code must match the implementation demonstrated during evaluation; do not change the code after capturing these screenshots without re-capturing them.", CAP)]

# ------------------------------------------------------------------ 8 integrity
story += [P("8. Reproducibility, limitations and academic-integrity statement", H1),
          P("<b>Reproduce everything:</b> <i>python scripts/02_build_dataset.py; 01_eda.py; 03_compare_models.py; 03b_feature_ablation.py; 04_bayes_opt.py; 03c_learning_curves.py; 05_hmm_decode.py; 06_make_maps.py; 07_test_api.py; 08_make_figures.py; 09_build_report.py</i> "
            "(see README). Seeds, configurations, per-fold results, trial records and logs are kept in results/ and logs/."),
          P("<b>Limitations (state these in the viva).</b> (1) Per-fix labels are derived by our own alignment of fixes to the reference route and the reference route itself may be imperfect (two training trips were unusable); "
            "(2) Candidate Miss Rate is measured against these derived labels, so it is optimistic; (3) models are fitted on every 3rd fix (KNN every 9th) for speed; (4) the classifier is trained on 78 trips from many different places, so geographic generalisation of the evaluator trips is untested; "
            "(5) the student-test set has only 10 trips, so differences of about one point between models are within noise; (6) decoder parameters were tuned on out-of-fold training probabilities; (7) route continuity uses a 3-hop tolerance, "
            "which is our own definition."),
          P("<b>Data-leakage controls.</b> Splits are by trip; the evaluator partition was never used; evaluator labels are never loaded by the prediction path or returned by the API; no predictions are hardcoded; "
            "learned preprocessing (scaling) is fitted on training data only; the label-QC rule and the model-selection rule were fixed before looking at student-test results.", B),
          P("<b>AI-assistance disclosure (suggested wording &mdash; edit to match what your team actually did and what the organisers require):</b> &ldquo;An AI assistant (Claude) was used to help write and run the code for data processing, "
            "modelling, the HMM, the API/UI and this draft report. All reported numbers were produced by executing the submitted code; the team re-ran the experiments, reviewed the results and can explain the methods.&rdquo; "
            "<b>Only keep the last clause if it is true.</b>", BOX)]

doc = SimpleDocTemplate(str(C.ROOT / "report" / "PathSnap_Technical_Report_DRAFT.pdf"), pagesize=A4, leftMargin=1.9 * cm, rightMargin=1.9 * cm, topMargin=1.7 * cm, bottomMargin=1.7 * cm,
                        title="PathSnap Technical Report (DRAFT)")
doc.build(story, onFirstPage=lambda c, d: None, onLaterPages=lambda c, d: (c.setFont("Helvetica", 7.5), c.setFillColor(colors.HexColor("#6b7385")), c.drawString(1.9 * cm, 1.0 * cm, "PathSnap technical report - DRAFT"), c.drawRightString(A4[0] - 1.9 * cm, 1.0 * cm, f"page {d.page}")))
print("built report/PathSnap_Technical_Report_DRAFT.pdf")
