# optimizer-lab

> **Ονομασία**: ο τελικός optimizer του project ονομάζεται **EchoMuon** = Muon +
> Temporal-Consistency Gating (κάθε singular direction του update εμπιστεύεται ανάλογα
> με την «ηχώ» της στο gradient history / slow momentum buffer) + memorization-gap
> controller (το gate κλιμακώνεται από ΜΕΤΡΗΜΕΝΟ overfitting: fresh vs re-evaluated
> recently-seen batches). Εργασιακά ονόματα κατά την ανάπτυξη: TCG, AutoTCG (v2) —
> στα ιστορικά bullets/run ids παραμένουν. Το παλαιότερο spectral-controller νήμα
> (εργασιακά «SpectralMuon») μετονομάστηκε σε **spectral-ctl** στα labels λόγω
> σύγκρουσης ονόματος με το SpecMuon (arXiv 2602.16167).

Πειραματικό testbed για δύο ιδέες πάνω σε Muon/SAM γεωμετρία:

- **Ιδέα 1 — ShrunkMuon («shrink, don't whiten»)**: αντί η ορθογωνοποίηση να στέλνει όλες τις
  singular values του momentum στο 1, εφαρμόζουμε Random-Matrix-Theory optimal shrinkage
  (Marchenko–Pastur noise estimate + Gavish–Donoho shrinker) ώστε κάθε singular direction να
  σταθμίζεται με την εμπιστοσύνη ότι είναι σήμα και όχι θόρυβος. Muon = όριο υψηλού SNR.
  **Πρόβλεψη**: ισοπαλία με Muon σε καθαρό regime (A), νίκη σε low-SNR regime (B).
- **Ιδέα 3 — Spectral recycling**: το singular spectrum του momentum (που ο Muon υπολογίζει
  έμμεσα και πετάει) καταγράφεται ως per-layer control signal· ελέγχουμε (α) αν προβλέπει
  loss spikes, (β) αν ένας απλός per-layer lr controller (stable-rank-based) βοηθά.

## Setup

- Μοντέλο: GPT 6L/6H/384d (~11M params), char-level Tiny Shakespeare, ctx 256, batch 64, bf16.
- Optimizers: `adamw` (όλα τα params), `muon` (hidden 2D matrices, AdamW στα υπόλοιπα),
  `shrunk` (ίδιο split, RMT-shrinkage αντί Newton–Schulz).
- Regimes: **A** = καθαρό. **B** = low SNR (Gaussian grad noise, σ = 1× per-tensor grad RMS).

## Fairness protocol (κριτήρια)

1. lr grid ανά optimizer (η κλίμακα update απορροφάται στο sweep) — warning αν το βέλτιστο
   πέσει σε άκρο του grid.
2. Ίδιο token budget, ίδιο μοντέλο, ίδια eval batches (σταθερό seed) για όλους.
3. Final: best lr × 3 seeds, mean ± std. Primary metric: **final val loss**. Secondary:
   best val, wall clock, tokens/s.
4. Σταθερά για όλους: grad clip 1.0, AdamW β=(0.9,0.95), wd 0.1 στα aux/AdamW params,
   momentum 0.95 + Nesterov στα Muon-class, aux lr 2e-3.

## Στάδια

```
docker run ... optlab smoke     # sanity, ~2-3 λεπτά
docker run ... optlab sweep     # pretraining protocol: lr grid × regimes
docker run ... optlab final     # pretraining protocol: best lr × 3 seeds (+ muon+ctl arm)
docker run ... optlab pretrain  # FT protocol: base Muon model σε enwik8 → results/base/
docker run ... optlab ft-sweep  # FT protocol: fine-tune lr grid × {F1 batch64, F2 batch8}
docker run ... optlab ft-final  # FT protocol: best lr × 3 seeds × 4 arms
docker run ... optlab analyze   # results/report.md + plots
docker run ... optlab all       # smoke → sweep → final → analyze
docker run ... optlab ft-all    # pretrain → ft-sweep → ft-final → analyze
```

Τα αποτελέσματα γράφονται στο mounted `results/` (report.md, plots/, runs/<id>/).

## Fine-tuning protocol (Ιδέα 1 repositioned)

Base: Muon-pretrained σε enwik8 (~1 epoch). Fine-tune σε byte-level Shakespeare (1M tokens,
ίδιο vocab 256) για 3000 steps — F1 (batch 64) ≈ 50 epochs, F2 (batch 8) ≈ 6 epochs. Arms:
muon, muon+wd 0.1 (ο δίκαιος regularizer-ανταγωνιστής), shrunk, adamw. Metrics: final &
best val στο Shakespeare (overfit_gap = final−best· 0 σημαίνει «δεν χρειάζεται early
stopping»), και enwik8 val στο τέλος (retention/catastrophic forgetting — το SNR-gate
προβλέπει λιγότερο ξέχασμα). Πρόβλεψη-κίνδυνος: στο v1-A (clean multi-epoch) το gate ΔΕΝ
προστάτεψε — ίσως χρειάζεται φυσικό gradient noise (F2/μικρό batch) για να πυροδοτηθεί.

## Συνέχιση μετά από crash / restart (--continue)

Όλα τα runs γράφουν atomic checkpoint (`runs/<id>/ckpt.pt`: model + optimizer + RNG states)
κάθε `--ckpt-every` steps (default 500). Αν κοπεί οτιδήποτε (crash, restart PC), απλώς
ξανατρέξε **την ίδια εντολή**: ολοκληρωμένα runs προσπερνιούνται (final.json), το
μισοτελειωμένο συνεχίζει από το τελευταίο checkpoint («RESUMING from step N» στο log).
Το detached container μπορεί να τρέξει και με `--restart on-failure:5` ώστε μετά από
reboot να συνεχίσει μόνο του (θέλει Docker Desktop autostart στα Windows settings).

## Αποτελέσματα (2026-08-02)

- **v1 (Shakespeare, ~100 epochs — confounded από overfitting)**: `results/v1_shakespeare/report.md`.
  Εύρημα: ο ShrunkMuon έχει ενσωματωμένο SNR-gate — σχεδόν ασυλία στο noise-memorization
  (final val 1.63 vs 3.38 του Muon στο noisy regime) και καλύτερο peak val και στα δύο regimes.
- **v2 (enwik8, <1 epoch — καθαρό πρωτόκολλο)**: `results/report.md`.
  Σε fresh-data καθεστώς ο ShrunkMuon ΔΕΝ κερδίζει σε ταχύτητα: χάνει από τον Muon και στα δύο
  regimes (A: 1.117 vs 1.080· B: 1.317 vs 1.168) — το momentum ήδη φιλτράρει το iid noise, και
  το gate κόβει και χρήσιμο σήμα. Το v1 πλεονέκτημα ήταν καθαρά η προστασία από memorization.
  **Ανατροπή στην Ιδέα 3**: ο stable-rank lr controller (muon+ctl) κέρδισε τον σκέτο Muon στο
  noisy regime (1.1591±0.004 vs 1.1676±0.004, καλύτερος και σε steps-to-target) με μηδενικό
  επιπλέον wall-clock — το πιο ελπιδοφόρο αποτέλεσμα του v2.

**Συμπέρασμα**: Ιδέα 1 → repositioning ως fine-tuning/overtraining regularizer (multi-epoch,
RL, small-data regimes), όχι pretraining speedup. Ιδέα 3 → ο controller αξίζει dedicated
follow-up με περισσότερα seeds.

- **v3 (fine-tuning protocol, enwik8-base → Shakespeare)**: `results/report.md` (FT sections).
  Το SNR-gate έχει **οριακή συνθήκη**: θέλει gradient noise για να πυροδοτηθεί. Στο F2
  (batch 8, θορυβώδη gradients) έκοψε το overfit gap του Muon 3× (0.052 vs 0.153) με ίδιο
  peak — δουλεύει όπως σχεδιάστηκε. Στο F1 (batch 64, καθαρά memorization gradients) ΔΕΝ
  προστάτεψε (gap 1.95). Το forgetting prediction ΑΠΕΤΥΧΕ: shrunk ξεχνάει όσο ο Muon.
  Και το σκληρό εύρημα: **σκέτος AdamW με μικρό tuned lr ισοφαρίζει ή κερδίζει τον
  ShrunkMuon σε όλα τα FT metrics** με 10× λιγότερο wall-clock. Το τίμιο positioning που
  απομένει: «Muon που δεν αυτοκαταστρέφεται όταν πέφτει το SNR» — σχετικό μόνο όπου ο
  Muon είναι υποχρεωτικός (συνεχόμενο pretraining→FT) ή σε RL (ανοιχτό, δεν το τεστάραμε).

- **v4 (Idea-3 controller: power + ablations, n=8)**: `results/report.md` (Welch table).
  Το κέρδος του controller ΕΠΙΒΕΒΑΙΩΘΗΚΕ στο regime B: 1.1601±0.003 vs 1.1697±0.003,
  **Welch t=6.45** (p≪0.01), ταχύτερος και σε steps-to-target· στο A ακριβώς no-op (t=0.00,
  ακίνδυνος). Ablations-σχολικό παράδειγμα: **inverse βλάπτει** (t=−5.20 → το σήμα είναι
  κατευθυντικό), **shuffled ουδέτερο** (t=−0.93 → μετράει η αντιστοίχιση layer↔scale, όχι
  η ποικιλία). Το `plots/ctl_scales.png` δείχνει τι έμαθε: σταθερό boost στα MLP
  down-projections (fc_proj), ελαφρύ φρένο αλλού — αυτόματη ανακάλυψη δομής τύπου
  Sharpness Disparity Principle από το φάσμα που ο Muon πετάει. Effect size ~0.01 nats
  σε 11M params με μηδενικό κόστος· ανοιχτό αν μεγαλώνει με το scale.

- **v5 (Idea-3 scale test, S/M/L)**: `results/report.md` (scale section, `plots/scale_delta.png`).
  Το ctl effect **επιβιώνει σε 10× scale-up**: paired Δ (ctl−muon) σημαντικό και στα τρία
  μεγέθη — S-11M: −0.0096 (t=−10.6, n=8)· M-38M: −0.0030 (t=−2.7, n=8)· L-114M: −0.0056
  (t=−12.1, n=6, το πιο συνεπές: sem 0.0005). Τα lr όλων των scales επαληθεύτηκαν ως
  εσωτερικά βέλτιστα (extra sweep runs στο 0.005 για M/L: και τα δύο χειρότερα από 0.01).
  Το μέγεθος του effect είναι μη-μονότονο (βουτιά στο M) — πραγματικό σχήμα, όχι lr
  artifact· ανοιχτό ερώτημα το γιατί. Claim: **persistence, όχι growth** (~0.3–0.8% του
  loss, μηδενικό κόστος, σε αντίθεση με το τυπικό «σβήσιμο» των optimizer gains στο scale).

- **v6–v7 (SpectralMuon v1→v2 + decisive probes)**: `results/report.md`. Τελική σύνθεση
  **SpectralMuon = Muon + rank + conf2(trend) + valve**, όλα από μία αμορτισμένη spectral
  μέτρηση. Πλήρης claims matrix: A clean ίσος/οριακά καλύτερος (1.0747 vs 1.0787)· B noisy
  καλύτερος (1.1594 vs 1.1697, t=+7.0)· C2 fault injection πολύ καλύτερος (1.1532 vs 1.2057,
  t=+12.1, valve attribution −0.0105)· D fine-tune σε tuned lr ΙΣΟΣ (1.3783 vs 1.3761,
  paired seeds) και σε λάθος lr δραματικά πιο robust. Probes που έκλεισαν τον μηχανισμό:
  το v1-conf όφελος στο D ήταν εξ ολοκλήρου effective-lr calibration (muonlow@0.001 gap
  0.0362 ≥ v1 0.0541)· η v1-conf ήττα στο B ΔΕΝ ήταν lr artifact (lr0.16 → 1.549, χειρότερα).
  Τίμια ταυτότητα: «ποτέ χειρότερος από tuned Muon· καλύτερος σε θόρυβο (scale-persistent
  έως 114M)· περιορίζει transient faults· αυτο-βαθμονομεί το lr» — με μηδενικό επιπλέον
  κόστος και κανένα νέο tuned hyperparameter.

- **v8 (natural-noise validation, regimes N/P)**: `results/report.md`. Το noise claim
  ΕΠΙΒΙΩΝΕΙ σε πραγματικό θόρυβο: N (batch 8, γνήσιος minibatch θόρυβος, 8000 steps):
  sm2 1.1829 vs muon 1.1885 (t=2.33)· P (enwik8 με 10% corrupted train bytes, καθαρό val):
  sm2 1.2269 vs muon 1.2318 (t=5.10). Ίδιο tuned lr (0.01) και στα δύο arms — όχι lr
  artifact. Εύρημα-κλειδί: στα natural regimes ο rank-only controller ΔΕΝ περνά μόνος το
  όριο σημαντικότητας (t=0.88/1.50)· μόνο η πλήρης σύνθεση rank+conf2+valve κερδίζει —
  τα components μαζί πετυχαίνουν ό,τι κανένα μόνο του. Effect size ~½ του synthetic
  (−0.005 vs −0.010 nats), τίμια αναφορά και των δύο στο write-up.

- **v9 (vision generalization: ViT/CIFAR-10, regimes VA/VP)**: `results/report.md`.
  Δεύτερη modality με πανομοιότυπο optimizer κώδικα (ViT με το ίδιο Block). VA clean:
  όλα τα arms ισόπαλα (paired t ≤1.0) — safety και εκτός LM. **VP (20% label noise):
  SpectralMuon2 +0.99% accuracy (74.86% vs 73.87%, paired t=+4.41, n=8)** και loss
  paired t=−2.25· rank-only ctl μόνο marginal (acc t=+1.83). Τρίτη επανάληψη του
  μοτίβου «η πλήρης σύνθεση κερδίζει εκεί που κανένα component μόνο του δεν φτάνει».
  Σημείωση στατιστικής: στο vision η seed variance είναι ~10× του LM — τα paired tests
  (ίδια seeds ανά arm) είναι το σωστό εργαλείο και πρέπει να μπουν στο analyze.py για
  το write-up.

- **v10 (breadth campaign: vision V100A/V100P/TIA/TIP + LM FA/FM)**: EchoMuon vs
  scheduled Muon, paired seeds, cosine παντού, δικά τους sweeps ανά arm (grid-edge
  extensions επιβεβαίωσαν όλα τα βέλτιστα εσωτερικά). Vision — CIFAR-100: +0.89%
  clean (t=+3.24) / +1.67% @20% noise (t=+4.70)· Tiny-ImageNet-200: +1.76% clean
  (t=+5.77) / +1.53% noisy (t=+4.69) — 6/6 vision κελιά νίκες σε 3 datasets. Vs tuned
  AdamW+cos: +8-17.5% παντού. LM modern benchmark — **FA (LLaMA-style 162M, RMSNorm/
  RoPE/SwiGLU, FineWeb-Edu + GPT-2 BPE): EchoMuon ΚΕΡΔΙΖΕΙ τον scheduled Muon −0.031
  nats (t=−6.32)**, αντιστροφή του enwik8-at-scale μοτίβου (πιθανή αιτία: φυσικός
  web-corpus θόρυβος + το BPE tail ζει στα embeddings, όχι στα Muon-managed matrices)·
  και οι δύο συντρίβουν τον AdamW (−0.19/−0.22 nats, t≈−33/−43). **FM (Mamba-2-style
  SSM 20L×512d): πρώτο κελί όπου ο AdamW κερδίζει τη Muon-οικογένεια (−0.042 nats,
  t=+4.1)** — το Muon πλεονέκτημα δεν μεταφέρεται αυτούσιο στη γεωμετρία του SSM
  (ετερογενές in_proj concat)· EchoMuon = Muon εκεί (t=−0.65, never-worse κρατά).
  Pre-registered αποκλίσεις που αναφέρονται: (α) το clean-vision κέρδος ΔΕΝ μικραίνει
  με το finer label space — μεγιστοποιείται στο Tiny (φυσική ασάφεια labels = γήπεδο
  του gate), (β) η FA νίκη ήταν απρόβλεπτη προς όφελός μας.

## Γνωστοί περιορισμοί

- Κλίμακα 11M params/char-level: αρκεί για το controlled ερώτημα SNR×γεωμετρία, ΔΕΝ αποδεικνύει
  scaling — τα ευρήματα θέλουν επιβεβαίωση σε μεγαλύτερο LM (βλ. Fantastic Pretraining Optimizers).
- Ο noise estimator υποθέτει ότι >50% του φάσματος είναι bulk (median-based) — ισχύει εδώ,
  θέλει έλεγχο σε άλλα regimes.
- Το regime B (συνθετικός θόρυβος) είναι proxy του RLVR low-SNR· επόμενο βήμα ένα πραγματικό
  RL fine-tune.
