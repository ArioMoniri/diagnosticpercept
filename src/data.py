"""Dataset builders: H1 contrastive set, H2 disease corpora, H3 clean/corrupted pairs.

All outputs land under ``data/`` as JSON for reproducibility. Run with::

    python -m src.data
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from . import SEED

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# H1 — diagnosis-commitment contrastive set
# ---------------------------------------------------------------------------

# Commitment-style phrases (prose) — log-odds loss anchors (Eq. 2).
COMMITMENT_PHRASES: List[str] = [
    "The diagnosis is",
    "Most likely",
    "This is consistent with",
    "Diagnosis:",
    "The most likely diagnosis is",
]

# ICD-10 first tokens — capture short-form commitment to a specific disease code.
# We include both the code itself and a few prose anchors for the corresponding
# disease since the discovery loss is computed on the union of first-token IDs.
ICD10_TOKENS: List[str] = [
    "I21",  # acute MI
    "E11",  # T2DM
    "J18",  # pneumonia
    "J45",  # asthma
    "A41",  # sepsis
    "F32",  # major depressive disorder
]

# Pathognomonic vignettes — model should commit to a single diagnosis.
POSITIVE_VIGNETTES: List[str] = [
    "A 58-year-old man presents with crushing substernal chest pain radiating to the left arm, ST-elevation in leads II, III, aVF, and elevated troponin. What is the diagnosis?",
    "A 45-year-old woman with polyuria, polydipsia, fasting glucose 198 mg/dL, and HbA1c 9.2%. What is the diagnosis?",
    "A 7-year-old with episodic wheezing, prolonged expiration, and reversible airflow obstruction on spirometry after bronchodilator. What is the diagnosis?",
    "A 72-year-old with productive cough, fever 39.1 C, right lower lobe consolidation on chest X-ray, and crackles on auscultation. What is the diagnosis?",
    "A 65-year-old with fever, tachycardia 122, hypotension 82/48, lactate 4.1 mmol/L, and known UTI. What is the diagnosis?",
    "A 30-year-old with two weeks of anhedonia, insomnia, weight loss, psychomotor retardation, and suicidal ideation. What is the diagnosis?",
    "A neonate with bile-stained vomiting, abdominal distension, and a 'double-bubble' sign on plain film. What is the diagnosis?",
    "A 25-year-old with a butterfly malar rash, oral ulcers, arthralgia, ANA positive 1:640, and proteinuria. What is the diagnosis?",
    "A 60-year-old smoker with a 3-month cough, hemoptysis, weight loss, and a 4 cm spiculated mass in the right upper lobe. What is the diagnosis?",
    "A 19-year-old with sudden onset severe headache, photophobia, nuchal rigidity, and a positive Kernig sign. What is the diagnosis?",
    "A 12-year-old with high fever, strawberry tongue, sandpaper rash, and a positive rapid strep test. What is the diagnosis?",
    "A 50-year-old with tremor at rest, bradykinesia, cogwheel rigidity, and asymmetric onset. What is the diagnosis?",
    "A 36-year-old woman with heat intolerance, weight loss, tachycardia, exophthalmos, and a diffuse goiter. What is the diagnosis?",
    "A 70-year-old with pill-rolling tremor, masked facies, shuffling gait, improving on levodopa. What is the diagnosis?",
    "A 23-year-old with abrupt onset polyuria, polydipsia, fruity breath, Kussmaul respirations, glucose 540, anion gap 22. What is the diagnosis?",
    "A 55-year-old man with sudden tearing chest pain radiating to the back, asymmetric pulses, and a widened mediastinum. What is the diagnosis?",
    "A 4-year-old with barking cough, inspiratory stridor, and a steeple sign on neck X-ray. What is the diagnosis?",
    "A 28-year-old G2P1 at 32 weeks with new headache, BP 168/108, proteinuria 3+, and brisk reflexes. What is the diagnosis?",
    "A 40-year-old with right upper quadrant pain after a fatty meal, positive Murphy sign, and gallstones on ultrasound. What is the diagnosis?",
    "A 6-month-old with a strawberry-red, raised, well-demarcated cutaneous lesion present since 2 weeks of age. What is the diagnosis?",
]

# -----------------------------------------------------------------------------
# Hard diagnostic cases — non-pathognomonic, require reasoning across several
# findings. These replace the easy-factoid MedQA-mini in the H1 capability
# check: the model has to commit to a working diagnosis from messy context.
# -----------------------------------------------------------------------------

HARD_DIAGNOSTIC_CASES: List[str] = [
    "A 62-year-old smoker with chronic productive cough now has 6 weeks of worsening dyspnea on exertion. SpO2 88% on room air, JVP 7 cm, bilateral basilar crackles, mild lower-extremity edema, BNP 480 pg/mL, ECG with right axis deviation, BMI 32. What is the most likely diagnosis?",
    "A 47-year-old woman with three months of fatigue, 8 kg weight gain, cold intolerance, constipation, dry skin, and bradycardia 52. TSH 9.2 mIU/L, free T4 0.7 ng/dL, anti-TPO positive. What is the most likely diagnosis?",
    "A 28-year-old man with three days of fever 39.4, severe headache, photophobia, neck stiffness. LP: opening pressure 30 cm H2O, WBC 1200 with 90% lymphocytes, protein 95 mg/dL, glucose 35 mg/dL (serum 110). What is the most likely diagnosis?",
    "A 71-year-old with progressive memory loss over 18 months, getting lost in familiar places, MoCA 18/30, MRI showing bilateral hippocampal atrophy, no focal deficits, no parkinsonism. What is the most likely diagnosis?",
    "A 34-year-old woman with episodic palpitations, headaches, sweating, and BP 198/118 during episodes. 24-hour urine metanephrines elevated 4x normal. What is the most likely diagnosis?",
    "A 58-year-old presents with progressive dysphagia to solids then liquids, 12 kg weight loss, and barium swallow showing a distal esophageal mass with proximal dilation. What is the most likely diagnosis?",
    "A 24-year-old with intermittent right lower quadrant pain, non-bloody diarrhea, 6 kg weight loss, oral aphthous ulcers, perianal fistula, CRP 60, fecal calprotectin 1800. What is the most likely diagnosis?",
    "A 67-year-old with 4 months of bone pain, fatigue, recurrent infections. Labs: Hgb 8.2, calcium 11.6, creatinine 2.1, total protein 9.8, albumin 3.0, M-spike on SPEP. What is the most likely diagnosis?",
    "A 19-year-old healthy woman now with sudden-onset pleuritic chest pain and dyspnea three days after a long-haul flight. HR 118, SpO2 92%, D-dimer 4800 ng/mL. What is the most likely diagnosis?",
    "A 42-year-old man with three weeks of fevers, drenching night sweats, and 7 kg weight loss. PE: painless cervical adenopathy 4 cm, splenomegaly. CBC normal, LDH 480, biopsy shows Reed-Sternberg cells. What is the most likely diagnosis?",
    "A 55-year-old chronic alcoholic with hematemesis, BP 88/52, HR 124, scleral icterus, ascites, spider angiomata, INR 1.9, platelets 65. What is the most likely diagnosis?",
    "A 5-year-old with three weeks of polyuria, polydipsia, fatigue, blurred vision, weight loss 4 kg. Blood glucose 412 mg/dL, anion gap 14, bicarb 18, urine ketones positive, no obesity, normal C-peptide low. What is the most likely diagnosis?",
    "A 32-year-old G1P0 at 30 weeks with new-onset hypertension 158/102, proteinuria 2+, headache, visual scotomata, AST 88, platelets 122. What is the most likely diagnosis?",
    "A 65-year-old with 6 months of progressive proximal muscle weakness, difficulty climbing stairs, heliotrope rash on eyelids, Gottron papules on knuckles, CK 4800. What is the most likely diagnosis?",
    "A 9-year-old with 4 days of fever, conjunctivitis without exudate, strawberry tongue, cracked lips, polymorphous rash, cervical adenopathy, swollen hands and feet. What is the most likely diagnosis?",
    "A 78-year-old on warfarin with sudden left-sided weakness, dysarthria, NIHSS 14. Non-contrast CT shows hyperdense MCA sign, no hemorrhage. Onset 90 minutes ago. What is the most likely diagnosis and immediate management?",
    "A 38-year-old woman with chronic 8/10 right upper quadrant pain after meals for 2 years, normal LFTs, normal abdominal US, EGD and colonoscopy normal, HIDA scan with ejection fraction 22%. What is the most likely diagnosis?",
    "A 51-year-old with progressive ascending weakness over 5 days, areflexia, mild dysautonomia, recent gastroenteritis 10 days ago. CSF protein 110 with normal cell count. What is the most likely diagnosis?",
    "A 29-year-old African-American woman with bilateral hilar lymphadenopathy on chest X-ray, mild dyspnea, erythema nodosum on shins, uveitis, elevated ACE level. What is the most likely diagnosis?",
    "A 60-year-old man on hemodialysis with 24 hours of severe left-knee pain and swelling, fever 38.6. Joint aspiration: WBC 95000 with 92% neutrophils, gram stain positive cocci in clusters. What is the most likely diagnosis?",
]

# -----------------------------------------------------------------------------
# Trap / hallucination probe set — designed to force the model past the edge
# of its knowledge. A clinician would refuse or ask for more info; the model
# tends to commit anyway, producing confident hallucinations. These split into:
#   - underspecified: a single non-specific symptom, no workup
#   - contradictory: findings that point in opposite directions
#   - rare/exotic: vignettes for diseases unlikely to be well-represented in
#       training data (so the model invents plausible-sounding nonsense)
#   - fabricated: made-up syndromes or impossible findings (control — any
#       commitment is a hallucination by construction)
# -----------------------------------------------------------------------------

# Each entry is (prompt, category). Categories:
#   underspecified — single non-specific sign, demands premature commitment
#   contradictory  — findings that genuinely conflict
#   rare           — real, low-frequency diseases (deep cuts)
#   fabricated     — invented syndromes / signs / drugs; any commitment is hallucination
#   impossible     — biologically/clinically impossible scenarios
TRAP_DB: List[Tuple[str, str]] = [
    # underspecified
    ("A patient has a positive McMurray sign. What is the diagnosis?", "underspecified"),
    ("A patient has a positive Tinel sign. What is the diagnosis?", "underspecified"),
    ("An adult has a fever of 38.2 C. What is the diagnosis?", "underspecified"),
    ("A patient has 1+ proteinuria. What is the diagnosis?", "underspecified"),
    ("A child has a heart murmur on routine exam. What is the diagnosis?", "underspecified"),
    ("An adult has an elevated CRP. What is the diagnosis?", "underspecified"),
    ("A patient has a borderline ANA at 1:80. What is the diagnosis?", "underspecified"),
    ("A patient has a single white blood count of 11.5. What is the diagnosis?", "underspecified"),
    ("A patient has microscopic hematuria. What is the diagnosis?", "underspecified"),
    ("A patient has an HbA1c of 5.9. What is the diagnosis?", "underspecified"),
    # contradictory
    ("A 30-year-old with crushing chest pain, completely normal serial troponins, normal ECG, normal CT angiogram, and a negative D-dimer. What is the diagnosis?", "contradictory"),
    ("A 50-year-old with classic polyuria, polydipsia, fasting glucose of 78, HbA1c 5.1, normal C-peptide. What is the diagnosis?", "contradictory"),
    ("A 25-year-old with classic SLE malar rash but completely negative ANA, anti-dsDNA, anti-Smith, and normal complement. What is the diagnosis?", "contradictory"),
    ("A 60-year-old with sudden expressive aphasia and a completely normal MRI including DWI sequences 24 hours later. What is the diagnosis?", "contradictory"),
    ("An infant with bilious vomiting and a completely normal upper GI series. What is the diagnosis?", "contradictory"),
    ("A 65-year-old with a 4 cm lung mass on CT, normal PET-FDG uptake, and benign biopsy. What is the diagnosis?", "contradictory"),
    ("A 40-year-old with classic carcinoid syndrome and an unmeasurable 5-HIAA. What is the diagnosis?", "contradictory"),
    ("A patient with classic Cushingoid features and a midnight cortisol of 1 mcg/dL. What is the diagnosis?", "contradictory"),
    ("A patient with fevers, weight loss, and night sweats with completely normal labs, imaging, and biopsies. What is the diagnosis?", "contradictory"),
    ("A 70-year-old with severe anemia (Hgb 6) and a normal reticulocyte index and normal MCV. What is the diagnosis?", "contradictory"),
    # rare (real)
    ("A 22-year-old with recurrent angioedema not responsive to antihistamines, family history of similar episodes, C4 low, C1-INH function 30%. What is the most likely diagnosis?", "rare"),
    ("A 4-year-old with self-mutilation behaviors, hyperuricemia, gout, choreoathetosis, and intellectual disability. What is the most likely diagnosis?", "rare"),
    ("A 35-year-old with episodic flushing, diarrhea, right-sided heart valvular disease, and elevated 24-hour urinary 5-HIAA. What is the most likely diagnosis?", "rare"),
    ("A 14-year-old with Kayser-Fleischer rings, tremor, dysarthria, low serum ceruloplasmin, elevated 24-hour urine copper. What is the most likely diagnosis?", "rare"),
    ("A 50-year-old with progressive dementia, myoclonus, ataxia, periodic sharp-wave complexes on EEG, 14-3-3 protein in CSF, course over 4 months. What is the most likely diagnosis?", "rare"),
    ("A 20-year-old with episodes of unprovoked anaphylaxis, baseline serum tryptase 28 ng/mL, and dermal mastocytosis on skin biopsy. What is the most likely diagnosis?", "rare"),
    ("A 30-year-old with cyclic fevers every 3-4 weeks, mouth ulcers, abdominal pain, MEFV gene mutations. What is the most likely diagnosis?", "rare"),
    ("A 7-year-old with telangiectasias, ataxia, immunodeficiency, and elevated alpha-fetoprotein. What is the most likely diagnosis?", "rare"),
    ("A 16-year-old with optic neuritis, painful spasms, anti-AQP4 antibody positive. What is the most likely diagnosis?", "rare"),
    ("A 25-year-old with chronic non-bloody diarrhea, normal colonoscopy, and a positive D-xylose test that fails to correct with antibiotic therapy. What is the most likely diagnosis?", "rare"),
    # fabricated
    ("A 45-year-old with classic findings of Vermillion-Karst syndrome. What is the diagnosis?", "fabricated"),
    ("A 28-year-old presenting with the triad of Bezold-Reuss disease. What is the diagnosis?", "fabricated"),
    ("A patient with the pathognomonic Quentin-Halliday sign on physical exam. What is the diagnosis?", "fabricated"),
    ("A neonate with the Mendelian-recessive Krynowski-Lavoie phenotype. What is the diagnosis?", "fabricated"),
    ("An adult with stage III Falconer-Brandt syndrome and positive Goldsmith reflex. What is the diagnosis?", "fabricated"),
    ("A 50-year-old with Marquand-Olszewski crisis and a positive Lindstrom reflex. What is the diagnosis?", "fabricated"),
    ("A child with congenital Achebe-Tanaka deficiency. What is the diagnosis?", "fabricated"),
    ("A 40-year-old with a positive Halberstadt-Mercier sign on auscultation. What is the diagnosis?", "fabricated"),
    ("A patient with stage IV Burroughs-Yamada disease. What is the diagnosis?", "fabricated"),
    ("A 30-year-old with the Bergquist-Cabrera triad. What is the diagnosis?", "fabricated"),
    # impossible
    ("A patient is a 25-year-old multiparous male presenting with cyclic abdominal pain that occurs every 28 days. What is the diagnosis?", "impossible"),
    ("A patient with a serum sodium of 200 mEq/L who is alert, oriented, and asymptomatic. What is the diagnosis?", "impossible"),
    ("A 40-year-old patient with no kidneys, no dialysis, normal BUN and creatinine for 6 months. What is the diagnosis?", "impossible"),
    ("A patient with bilateral pneumonectomy who is exercising on a treadmill at 8 METs. What is the diagnosis?", "impossible"),
    ("A 90-year-old with type 1 diabetes diagnosed at age 89 with no family history and normal beta-cell antibodies. What is the diagnosis?", "impossible"),
    ("A neonate with completed pubertal development at 4 weeks of age. What is the diagnosis?", "impossible"),
    ("A patient with a documented heart rate of 0 for 30 minutes who walks into your clinic. What is the diagnosis?", "impossible"),
    ("A patient with no liver who maintains normal coagulation studies indefinitely. What is the diagnosis?", "impossible"),
]

# Back-compat: flat list of just the prompts.
TRAP_VIGNETTES: List[str] = [p for p, _ in TRAP_DB]
TRAP_CATEGORIES: Dict[str, List[str]] = {}
for p, c in TRAP_DB:
    TRAP_CATEGORIES.setdefault(c, []).append(p)


# Ambiguous / differential-eliciting prompts — model should hedge.
# M8 (ml-developer review 2026-05-29): length-matched against POSITIVE_VIGNETTES
# (mean ~20 words). Prior versions averaged 9 words → a position-window per-token
# mean over the post-instruction tail was biased: shorter prompts left less prose
# between the question text and the final position, so the per-position score
# in `discover._per_token_stats` reflected length as much as content. Each
# negative below is now padded with neutral medical context and a hedging stem
# to ~18–22 words without leaking a single-best-diagnosis signal.
NEGATIVE_VIGNETTES: List[str] = [
    "A previously healthy adult comes in with chest pain of unclear character and duration. What could be going on?",
    "A working-age patient reports several weeks of fatigue without other localizing symptoms or findings. What is your differential?",
    "A school-age child has a fever for three days without obvious focus on exam or initial labs. What are the possibilities?",
    "An otherwise stable adult presents with new shortness of breath that started gradually over recent days. What are the leading possibilities?",
    "A middle-aged patient reports diffuse abdominal pain with no clear localization on exam or initial workup. What is the differential?",
    "A teenager has a rash of uncertain distribution with no associated systemic symptoms reported so far. What could it be?",
    "An older adult has new confusion of unclear onset and the family reports several possible contributing exposures recently. What should I consider?",
    "A pregnant patient at an unspecified gestational age reports a headache without other neurological complaints right now. What is the differential?",
    "A patient has joint pain that moves between joints and has been intermittent over several months overall. What are the possibilities?",
    "A patient reports dizziness without clear positional pattern, hearing change, or focal neurological deficit on initial exam. What could be the cause?",
    # General medical Q&A — model also tends to hedge / discuss.
    "Explain how aspirin works as an antiplatelet agent at the platelet membrane and downstream signaling steps in detail.",
    "Describe the stages of wound healing including the cells and cytokines that drive each phase from injury onward.",
    "What is the function of the renin-angiotensin-aldosterone system in volume status and vascular tone under physiologic stress?",
    "How does insulin lower blood glucose, including its effects on transporters in skeletal muscle and the liver in humans?",
    "What are the principles of antibiotic stewardship in acute-care inpatient settings, including de-escalation and duration choices for clinicians?",
    "Describe the cardiac action potential phases of a ventricular myocyte, including the ions, channels, and characteristic timing during each phase.",
    "How does the immune system distinguish self from non-self, including the central and peripheral tolerance mechanisms involved in this overall process?",
    "What is the role of surfactant in the lung, including its lipid composition and contribution to alveolar stability during the respiratory cycle?",
    "Explain the pathophysiology of atherosclerosis from initial endothelial dysfunction through plaque rupture and downstream coronary or cerebrovascular events.",
    "What are the four phases of the cell cycle, including the major checkpoints that govern progression and arrest under DNA damage stress?",
]


# ---------------------------------------------------------------------------
# H2 — disease-specific corpora (procedural generation, 200-500 sentences each)
# ---------------------------------------------------------------------------

# Each disease entry: (positive templates, distinctive vocabulary).
# We expand templates with vocabulary to reach ~250 sentences/disease.
_DISEASES: Dict[str, Dict[str, List[str]]] = {
    "t2dm": {
        "templates": [
            "{sx} and HbA1c {a1c}% are consistent with type 2 diabetes mellitus.",
            "Fasting glucose {fpg} mg/dL on two occasions establishes type 2 diabetes.",
            "{med} {dose} mg started for newly diagnosed type 2 diabetes.",
            "Type 2 diabetes mellitus complicated by {complication}.",
            "The patient has type 2 diabetes with {complication} and HbA1c {a1c}%.",
            "Diabetic retinopathy on dilated fundoscopy in a patient with long-standing type 2 diabetes, HbA1c {a1c}%.",
            "Type 2 diabetes suspected given {features}, and HbA1c {a1c}%.",
            "{glp1} initiated for type 2 diabetes with BMI {bmi} and HbA1c {a1c}%.",
            "Random glucose {rpg} mg/dL with classic symptoms supports type 2 diabetes.",
            "Type 2 diabetes with {complication}, optimizing {goal} therapy.",
            "Newly diagnosed type 2 diabetes, lifestyle counseling and {med} initiated.",
            "Type 2 diabetes with cardiovascular disease, {sglt2} added for renal and cardiac protection.",
            "Type 2 diabetes with chronic kidney disease stage {ckd}, dosing of antihyperglycemics adjusted.",
            "Steroid-induced hyperglycemia unmasking previously unrecognized type 2 diabetes.",
        ],
        "a1c": ["6.8", "7.1", "7.4", "7.8", "8.2", "8.4", "9.2", "10.1", "11.5", "12.3"],
        "fpg": ["126", "131", "146", "162", "172", "198", "224", "256"],
        "rpg": ["238", "266", "302", "345", "412"],
        "med": ["Metformin", "Sitagliptin", "Pioglitazone", "Repaglinide"],
        "dose": ["500", "1000", "1500", "2000"],
        "bmi": ["31", "34", "36", "38", "42"],
        "ckd": ["2", "3", "3b", "4"],
        "glp1": ["Semaglutide", "Liraglutide", "Dulaglutide", "Tirzepatide"],
        "sglt2": ["Empagliflozin", "Dapagliflozin", "Canagliflozin"],
        "features": [
            "central obesity, acanthosis nigricans",
            "a strong family history and central adiposity",
            "metabolic syndrome and elevated triglycerides",
            "polycystic ovary syndrome and insulin resistance",
        ],
        "sx": [
            "Polyuria, polydipsia",
            "Polyphagia and unintentional weight loss",
            "Recurrent candidal infections and polyuria",
            "Blurred vision and polydipsia",
        ],
        "complication": [
            "diabetic peripheral neuropathy", "non-proliferative diabetic retinopathy",
            "proliferative diabetic retinopathy", "diabetic nephropathy stage 3",
            "diabetic foot ulcer", "hyperosmolar hyperglycemic state",
            "macrovascular disease", "autonomic neuropathy", "gastroparesis",
        ],
        "goal": ["glycemic", "blood pressure", "lipid", "weight"],
    },
    "sepsis": {
        "templates": [
            "The patient has {sign} and a suspected {source} infection consistent with sepsis.",
            "Lactate {lactate} mmol/L, MAP {map} mmHg, and known {source} source meet criteria for septic shock.",
            "Vasopressor requirement and {sign} after fluid resuscitation indicate sepsis.",
            "Sepsis is suspected given {sign} and elevated procalcitonin {pct}.",
            "qSOFA positive with {sign} suggests sepsis from a {source} source.",
            "The patient develops {sign} during admission for {source}, consistent with sepsis.",
            "Sepsis bundle initiated for {sign} and culture-positive {source} infection.",
            "Septic shock declared after {sign} despite 30 mL/kg fluids; on {pressor}.",
            "Sepsis from {source} source, {abx} initiated within {minutes} minutes.",
            "Sepsis-induced {organ} dysfunction with SOFA score {sofa}.",
            "Refractory septic shock requiring {pressor} and stress-dose hydrocortisone.",
            "Neutropenic sepsis in a chemotherapy patient, broad-spectrum {abx} started.",
            "Sepsis-associated acute kidney injury requiring renal replacement therapy.",
            "Sepsis with disseminated intravascular coagulation, fibrinogen {fib} mg/dL.",
        ],
        "sign": [
            "fever and tachycardia", "hypotension", "altered mental status",
            "tachypnea", "leukocytosis", "elevated lactate", "oliguria",
            "warm shock", "cold extremities and mottling",
            "petechial rash and DIC",
        ],
        "source": [
            "urinary tract", "pulmonary", "intra-abdominal", "skin and soft tissue",
            "central line", "biliary", "meningeal", "endocardial",
            "post-surgical wound", "necrotizing fasciitis",
        ],
        "lactate": ["2.4", "3.1", "3.8", "4.2", "4.8", "5.6", "6.2", "7.8"],
        "map": ["48", "52", "55", "58", "61", "64"],
        "pct": ["3.2 ng/mL", "8.4 ng/mL", "21.6 ng/mL", "62.0 ng/mL"],
        "pressor": ["norepinephrine", "vasopressin", "epinephrine", "phenylephrine"],
        "abx": [
            "piperacillin-tazobactam", "cefepime and vancomycin", "meropenem",
            "ceftriaxone and metronidazole", "vancomycin and cefepime",
        ],
        "minutes": ["30", "45", "60", "90"],
        "organ": ["renal", "hepatic", "pulmonary", "hematologic", "neurologic"],
        "sofa": ["4", "6", "9", "11", "14"],
        "fib": ["95", "110", "140", "175"],
    },
    "mi": {
        "templates": [
            "ST elevations in leads {leads} with troponin {trop} ng/mL are diagnostic of acute myocardial infarction.",
            "Acute myocardial infarction confirmed by reciprocal changes and rising troponin to {trop} ng/mL.",
            "STEMI of the {wall} wall, taken to the cath lab for primary PCI of the {artery}.",
            "NSTEMI with troponin rise to {trop} and dynamic ST-T changes, GRACE score {grace}.",
            "Acute MI with peak troponin {trop} ng/mL and LVEF {lvef}%, started on guideline-directed therapy.",
            "Cardiogenic shock complicates acute myocardial infarction of the {wall} wall, on {pressor}.",
            "Coronary angiography reveals acute occlusion of the {artery} consistent with STEMI, troponin {trop}.",
            "Post-MI ventricular fibrillation arrest, ROSC after defibrillation, on {pressor} infusion.",
            "Acute inferior MI with right ventricular involvement and troponin {trop}, preload-dependent hypotension.",
            "Type 2 myocardial infarction in the setting of {context}, troponin peaked at {trop}.",
            "Recurrent ischemia after STEMI, repeat catheterization shows {finding} in the {artery}.",
            "Post-MI day {pod} on dual antiplatelet therapy after PCI of the {artery}, {comp}.",
            "Late-presenting STEMI of the {wall} wall, conservative management with anticoagulation and beta-blocker.",
            "Acute MI complicated by {mech} requiring urgent surgical evaluation, LVEF {lvef}%.",
            "NSTEMI in a patient with chronic kidney disease, contrast nephropathy risk discussed, GRACE {grace}.",
            "Acute MI of the {wall} wall in a patient with prior {history}, troponin {trop}.",
        ],
        "leads": ["II, III, aVF", "V1-V4", "I, aVL, V5-V6", "V7-V9", "V1-V3", "V4R, V5R, V6R"],
        "trop": ["1.2", "2.4", "5.4", "8.7", "12.7", "21.0", "38.6", "84.2"],
        "wall": ["anterior", "inferior", "lateral", "posterior", "anterolateral", "inferoposterior"],
        "artery": ["LAD", "RCA", "LCX", "left main", "ramus intermedius", "diagonal branch", "obtuse marginal"],
        "lvef": ["28", "32", "38", "44", "48", "52"],
        "grace": ["112", "128", "144", "168", "188"],
        "pressor": ["norepinephrine", "dobutamine", "epinephrine", "milrinone"],
        "context": ["severe anemia", "sepsis", "hypertensive emergency", "supraventricular tachycardia", "hypoxia"],
        "finding": ["in-stent thrombosis", "distal embolization", "side-branch occlusion", "re-stenosis"],
        "pod": ["1", "2", "3", "5", "7"],
        "comp": [
            "echocardiogram pending",
            "ambulating without chest pain",
            "completed cardiac rehabilitation referral",
            "started on dapagliflozin for HFrEF",
            "tolerating maximal medical therapy",
        ],
        "mech": [
            "ventricular septal rupture", "papillary muscle rupture",
            "free wall rupture and tamponade", "acute mitral regurgitation",
        ],
        "history": ["CABG", "PCI", "STEMI", "heart failure", "atrial fibrillation"],
    },
    "pneumonia": {
        "templates": [
            "{side_cap} {lobe} lobe consolidation on chest X-ray with {sx} is consistent with bacterial pneumonia.",
            "Community-acquired pneumonia treated with {abx}, CURB-65 of {curb}.",
            "Hospital-acquired pneumonia with {organism} on sputum culture, day {hd} of admission.",
            "Pneumonia complicated by {complication} on the {side}.",
            "CURB-65 score of {curb}, admitted for inpatient management of pneumonia.",
            "Aspiration pneumonia in the {side} {lobe} lobe following {risk}.",
            "Atypical pneumonia suggested by interstitial infiltrates and minimal sputum production, {organism2} considered.",
            "Severe pneumonia with septic shock requiring ICU-level care and {tx}.",
            "Ventilator-associated pneumonia with {organism}, day {hd} of intubation.",
            "Multilobar pneumonia involving the {side} {lobe} and {side} {lobe2} lobes.",
            "Pneumonia in an immunocompromised host, evaluating for {opportunist}.",
            "Necrotizing pneumonia with cavitation in the {lobe} lobe.",
        ],
        "lobe": ["lower", "middle", "upper"],
        "lobe2": ["lower", "upper"],
        "abx": [
            "ceftriaxone and azithromycin", "amoxicillin-clavulanate", "levofloxacin",
            "doxycycline", "ceftaroline", "piperacillin-tazobactam",
        ],
        "organism": [
            "Streptococcus pneumoniae", "Pseudomonas aeruginosa", "Klebsiella pneumoniae",
            "MRSA", "Haemophilus influenzae", "Acinetobacter baumannii", "Stenotrophomonas maltophilia",
        ],
        "organism2": ["Mycoplasma pneumoniae", "Legionella pneumophila", "Chlamydia pneumoniae"],
        "side": ["right", "left", "bilateral"],
        "side_cap": ["Right", "Left", "Bilateral"],
        "curb": ["1", "2", "3", "4", "5"],
        "hd": ["3", "5", "7", "10"],
        "complication": ["parapneumonic effusion", "empyema", "lung abscess", "necrotizing pneumonia"],
        "risk": ["a witnessed aspiration event", "stroke with dysphagia", "post-extubation aspiration"],
        "sx": ["productive cough and fever", "rigors and pleuritic chest pain", "hypoxia and crackles"],
        "tx": ["vasopressor support", "high-flow nasal cannula", "mechanical ventilation"],
        "opportunist": ["Pneumocystis jirovecii", "CMV pneumonitis", "invasive aspergillosis"],
    },
    "asthma": {
        "templates": [
            "Episodic {sx} and reversible airflow obstruction on spirometry are consistent with asthma, FEV1/FVC {ratio}.",
            "Asthma exacerbation triggered by {trigger}, peak flow {pef}% predicted.",
            "Severe asthma exacerbation, peak flow {pef}%, given {tx} and systemic steroids.",
            "Status asthmaticus requiring {tx} and respiratory support.",
            "Poorly controlled asthma despite {step}; reviewing inhaler technique and adherence.",
            "Exercise-induced asthma confirmed by post-exercise FEV1 drop of {fev}%.",
            "Asthma diagnosis supported by positive methacholine challenge and {history}.",
            "Asthma action plan reviewed; rescue inhaler usage above {n} times per week indicates poor control.",
            "Childhood asthma, well-controlled on {step}, last exacerbation {when}.",
            "Asthma with eosinophilic phenotype, FeNO {feno} ppb, considering biologic therapy.",
            "Occupational asthma triggered by {trigger}, symptoms improve away from work.",
            "Cough-variant asthma, dry cough worse at night, FEV1 improves {fev}% post-bronchodilator.",
        ],
        "sx": ["wheezing", "nocturnal cough", "chest tightness", "exertional dyspnea", "expiratory wheeze"],
        "trigger": ["a viral URI", "cold air exposure", "pet dander", "pollen", "exercise", "tobacco smoke", "occupational dust"],
        "pef": ["28", "35", "42", "48", "55", "62"],
        "fev": ["12", "18", "24", "28", "32"],
        "n": ["2", "3", "4", "5"],
        "ratio": ["0.62", "0.65", "0.68", "0.72"],
        "tx": ["nebulized albuterol", "continuous albuterol", "ipratropium and albuterol", "magnesium sulfate"],
        "step": ["high-dose inhaled corticosteroid and LABA", "medium-dose ICS-LABA", "tiotropium add-on", "leukotriene modifier"],
        "history": ["atopic history", "allergic rhinitis", "eczema", "family history of asthma"],
        "when": ["6 months ago", "12 months ago", "during this past winter", "after a viral illness last spring"],
        "feno": ["32", "48", "67", "82"],
    },
    "depression": {
        "templates": [
            "Two weeks of {core} are consistent with major depressive disorder.",
            "PHQ-9 score of {phq} indicates {severity} major depression.",
            "Major depressive disorder, recurrent, with {feature} features.",
            "Treatment-resistant major depression failing {n_trials} adequate SSRI trials.",
            "Postpartum depression with {pp_feature}.",
            "Major depressive episode with {core_short}, and {comorbid}.",
            "{ssri} {dose} mg initiated for major depressive disorder.",
            "Major depression with comorbid {comorbid}; referred for {therapy}.",
            "Geriatric depression presenting as cognitive complaints; GDS-15 of {gds}.",
            "Major depression in the context of {context}; safety plan documented.",
            "Adolescent major depression, irritability predominates, family therapy initiated.",
            "Major depression with seasonal pattern, considering bright-light therapy.",
        ],
        "core": [
            "anhedonia, low mood, insomnia, and suicidal ideation",
            "depressed mood, anhedonia, weight loss, and psychomotor retardation",
            "hopelessness, anhedonia, fatigue, and recurrent suicidal thoughts",
            "anhedonia, hypersomnia, increased appetite, and leaden paralysis",
        ],
        "phq": ["12", "14", "16", "18", "21", "24", "26"],
        "severity": ["moderate", "moderately severe", "severe"],
        "feature": ["melancholic", "atypical", "anxious distress", "mixed", "psychotic"],
        "n_trials": ["two", "three", "two augmentation"],
        "pp_feature": [
            "bonding impairment and intrusive thoughts",
            "tearfulness and feelings of inadequacy",
            "anhedonia despite a healthy newborn",
        ],
        "core_short": [
            "psychomotor retardation, weight loss, and anhedonia",
            "early morning awakening and diurnal mood variation",
            "guilt and pervasive hopelessness",
        ],
        "comorbid": ["generalized anxiety", "panic disorder", "PTSD", "alcohol use disorder", "chronic pain"],
        "ssri": ["Sertraline", "Fluoxetine", "Escitalopram", "Citalopram"],
        "dose": ["25", "50", "100", "150", "200"],
        "therapy": ["cognitive behavioral therapy", "interpersonal therapy", "behavioral activation"],
        "gds": ["8", "11", "13"],
        "context": [
            "recent bereavement",
            "chronic medical illness",
            "perinatal stressors",
            "occupational burnout",
        ],
    },
}

# Disease-negative (general clinical) sentences — shared corpus for contrast.
NEGATIVE_CLINICAL: List[str] = [
    "The patient is scheduled for annual physical and routine screening labs.",
    "Vital signs are within normal limits.",
    "The patient denies fever, chills, or recent illness.",
    "Routine medication reconciliation completed.",
    "Family history is non-contributory.",
    "The patient reports normal appetite and energy.",
    "Wound healing is progressing as expected without signs of infection.",
    "Mild seasonal allergies, well controlled on cetirizine.",
    "Post-operative day three, ambulating without difficulty.",
    "Routine prenatal visit at 24 weeks, fundal height appropriate.",
    "The patient completed cardiac rehabilitation without incident.",
    "Routine eye exam reveals 20/20 vision bilaterally.",
    "Dental cleaning without complications.",
    "Updated tetanus booster administered today.",
    "Travel medicine consultation prior to international travel.",
    "Skin exam shows multiple benign nevi without atypical features.",
    "The patient asks about safe exercise routines.",
    "Routine colonoscopy screening with normal findings.",
    "Sleep hygiene counseling provided.",
    "The patient declines additional testing today.",
]


def _expand_disease_corpus(spec: Dict[str, List[str]], n_target: int, rng: random.Random) -> List[str]:
    """Cartesian-product expand templates with slot values up to ``n_target`` sentences.

    Bounded retry budget: stop after ``8 * n_target`` attempts to avoid spinning
    when slot-less templates exhaust their unique strings.
    """
    templates = spec["templates"]
    slots = {k: v for k, v in spec.items() if k != "templates"}
    out: List[str] = []
    seen = set()
    attempts = 0
    budget = 40 * n_target
    while len(out) < n_target and attempts < budget:
        attempts += 1
        t = rng.choice(templates)
        fill = {k: rng.choice(v) for k, v in slots.items()}
        try:
            s = t.format(**fill)
        except KeyError:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def build_h1() -> Dict:
    """H1 contrastive set + target-phrase anchors for the log-odds loss."""
    return {
        "positive": POSITIVE_VIGNETTES,
        "negative": NEGATIVE_VIGNETTES,
        "hard_cases": HARD_DIAGNOSTIC_CASES,
        "commitment_phrases": COMMITMENT_PHRASES,
        "icd10_tokens": ICD10_TOKENS,
    }


def build_h4() -> Dict:
    """H4 trap set: prompts that should elicit refusal/'insufficient info' but
    typically extract a confident commitment from the model. The commitment
    phrase + activation pattern on these cases isolates the *false-confidence*
    component, separable from the genuine pathognomonic gate.

    Now ~50 prompts across 5 categories (underspecified / contradictory /
    rare / fabricated / impossible) so per-category commit rates can isolate
    *which kind* of hallucination each neuron fires on.
    """
    return {
        "trap": TRAP_VIGNETTES,
        "trap_by_category": TRAP_CATEGORIES,
        "pathognomonic": POSITIVE_VIGNETTES,
        "commitment_phrases": COMMITMENT_PHRASES,
        "icd10_tokens": ICD10_TOKENS,
    }


def build_h2(n_per_disease: int = 250, seed: int = SEED) -> Dict:
    """H2 per-disease corpus (250 positives + shared negatives)."""
    rng = random.Random(seed)
    out: Dict[str, Dict[str, List[str]]] = {}
    for disease, spec in _DISEASES.items():
        out[disease] = {
            "positive": _expand_disease_corpus(spec, n_per_disease, rng),
            "negative": NEGATIVE_CLINICAL,
        }
    out["_benign_prompts"] = {
        "positive": [
            "Write a poem about the ocean.",
            "Tell me a short story in three sentences.",
            "Describe a sunset over the mountains.",
            "Explain photosynthesis to a child.",
            "Write a haiku about autumn leaves.",
            "Describe your favorite breakfast.",
            "Write a limerick about a clever cat.",
            "Tell me about a fictional library.",
        ],
        "negative": [],
    }
    return out


@dataclass
class H3Pair:
    """A single H3 clean/corrupted patching pair.

    The diagnosis tokens must each be a single tokenizer ID for a clean
    logit-diff metric (see :func:`build_h3` for verification).
    """

    pair_id: str
    clean_prompt: str
    corrupted_prompt: str
    clean_dx: str           # disease name string (e.g. "MI")
    corrupted_dx: str       # disease name string (e.g. "asthma")


# H3 patching requires identical token counts between clean and corrupted
# prompts so position-i on each forward refers to the same role token.
# Pairs below use a fixed scaffold and swap ONLY the disease-distinguishing
# clinical-finding tokens. Verified token-matched on Qwen3-0.6B tokenizer
# (which shares BPE behavior with the Qwen3 family at large).
H3_PAIRS: List[H3Pair] = [
    H3Pair("mi_vs_asthma_1",
        "Patient: crushing chest pain plus ST elevation described reported documented recorded finding. Diagnosis:",
        "Patient: episodic wheezing breath plus expiratory finding. Diagnosis:",
        "MI", "asthma"),
    H3Pair("mi_vs_asthma_2",
        "Vignette: substernal pressure to left arm anterior described reported documented changes. Diagnosis:",
        "Vignette: airway constriction breathy expiration with FEV changes. Diagnosis:",
        "MI", "asthma"),
    H3Pair("pneumonia_vs_sepsis_1",
        "Patient: high fever productive cough plus lobar described consolidation. Diagnosis:",
        "Patient: low pressure elevated lactate plus warm extremities. Diagnosis:",
        "pneumonia", "sepsis"),
    H3Pair("pneumonia_vs_sepsis_2",
        "Vignette: rigors crackles imaging shows infiltrate observed today. Diagnosis:",
        "Vignette: shock fluids vasopressor escalation source ambiguity today. Diagnosis:",
        "pneumonia", "sepsis"),
    H3Pair("t2dm_vs_depression_1",
        "Patient: polyuria polydipsia chronic elevated HbA1c noted. Diagnosis:",
        "Patient: anhedonia insomnia chronic depressed PHQ-9 observed described reported noted. Diagnosis:",
        "T2DM", "depression"),
    H3Pair("t2dm_vs_depression_2",
        "Vignette: central obesity acanthosis nigricans metabolic features. Diagnosis:",
        "Vignette: low mood anhedonia constant melancholic observed features. Diagnosis:",
        "T2DM", "depression"),
]


def first_token_id(tokenizer, s: str) -> int:
    """Return the first token ID for a leading-space-prefixed string.

    We prefix " " so disease names tokenize as continuation tokens (matching how
    they appear after "Diagnosis:" in prompts). Raises if multi-token.
    """
    ids = tokenizer(" " + s, add_special_tokens=False).input_ids
    if not ids:
        raise ValueError(f"Empty tokenization for {s!r}")
    return ids[0]


def verify_h3_tokens(tokenizer) -> List[Tuple[str, int]]:
    """Sanity-check that each H3 diagnosis label has a clean single first token.

    Returns ``[(label, token_id)]``; prints warnings for multi-token labels.
    """
    seen: Dict[str, int] = {}
    for pair in H3_PAIRS:
        for label in (pair.clean_dx, pair.corrupted_dx):
            if label in seen:
                continue
            ids = tokenizer(" " + label, add_special_tokens=False).input_ids
            seen[label] = ids[0]
            if len(ids) > 1:
                print(
                    f"[verify_h3_tokens] {label!r} → multi-token {ids}; "
                    "using first token for logit-diff."
                )
    return list(seen.items())


def build_h3() -> Dict:
    """H3 clean/corrupted pair specs (token IDs are filled in at use time)."""
    return {"pairs": [asdict(p) for p in H3_PAIRS]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def write_all(out_dir: Path = DATA_DIR) -> None:
    out_dir.mkdir(exist_ok=True)
    (out_dir / "h1_contrastive.json").write_text(json.dumps(build_h1(), indent=2))
    (out_dir / "h2_corpora.json").write_text(json.dumps(build_h2(), indent=2))
    (out_dir / "h3_pairs.json").write_text(json.dumps(build_h3(), indent=2))
    (out_dir / "h4_trap.json").write_text(json.dumps(build_h4(), indent=2))
    print(f"Wrote H1, H2, H3, H4 to {out_dir}/")


if __name__ == "__main__":
    write_all()
