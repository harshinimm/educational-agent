"""One-shot synthetic data seeder for the O-Level Chemistry demo.

Run locally before recording the demo / before deploying:

    python seed_data.py

Populates:
  - data/concepts.csv     — the O-Level Chemistry syllabus concept list
  - data/flashcards.csv   — a couple of manual flashcards per concept
  - data/notes.csv        — short reference-note chunks per concept (the
                            corpus retriever.py searches)
  - the `interactions` table — ~5 weeks of simulated study history, one
    hidden BKT-style knowledge process per concept, so BKT/DKT/the review
    queue/the tutor policy all have real signal to work with.

Not wired into the Streamlit app — this is a dev-time script, re-run it
any time you want a clean demo dataset (it overwrites existing content).
"""
import csv
import random
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import db

DATA_DIR = Path(__file__).parent / "data"

# concept -> difficulty tier (drives simulated P(L0) / learn rate below)
CONCEPTS = {
    "Kinetic Particle Theory": "easy",
    "Atomic Structure": "medium",
    "Chemical Bonding - Ionic": "medium",
    "Chemical Bonding - Covalent": "hard",
    "Chemical Bonding - Metallic": "easy",
    "Formulae and Equations": "medium",
    "The Mole Concept": "hard",
    "Stoichiometry Calculations": "hard",
    "Acids and Bases": "medium",
    "Salts and Preparation of Salts": "medium",
    "Qualitative Analysis (Tests for Ions and Gases)": "easy",
    "Redox Reactions": "hard",
    "Electrolysis": "hard",
    "Energy Changes": "medium",
    "Rate of Reaction": "medium",
    "Reversible Reactions and Equilibrium": "hard",
    "Periodic Table Trends": "easy",
    "Metals and the Reactivity Series": "medium",
    "Extraction of Metals": "medium",
    "Chemistry of Air and the Environment": "easy",
    "Chemistry of Water": "easy",
    "Organic Chemistry - Alkanes and Alkenes": "medium",
    "Organic Chemistry - Alcohols and Carboxylic Acids": "medium",
    "Polymers": "easy",
    "Experimental Design and Techniques": "medium",
}

# difficulty tier -> (P_L0 prior knowledge, P_T learn rate per opportunity)
DIFFICULTY_PARAMS = {
    "easy": (0.35, 0.35),
    "medium": (0.20, 0.22),
    "hard": (0.08, 0.07),
}
# fewer opportunities for hard concepts too, so several genuinely stay
# under-mastered by the end of the log — real variance for the review
# queue / weak-area selection to work with, not everything converging to 1.0
ATTEMPTS_RANGE = {
    "easy": (15, 22),
    "medium": (12, 18),
    "hard": (8, 14),
}
P_SLIP = 0.10
P_GUESS = 0.20

NOTES = {
    "Kinetic Particle Theory": "Matter exists as solid, liquid, or gas depending on the arrangement and energy of its particles. Heating increases particle kinetic energy, weakening the forces holding particles together and driving melting, evaporation, and diffusion.",
    "Atomic Structure": "Atoms consist of protons and neutrons in a central nucleus, surrounded by electrons in shells. Proton number defines the element; nucleon number is protons plus neutrons. Isotopes share proton number but differ in neutron number.",
    "Chemical Bonding - Ionic": "Ionic bonds form by electron transfer between a metal and non-metal, producing oppositely charged ions held together by strong electrostatic attraction in a giant lattice. This gives high melting points and conductivity only when molten or dissolved.",
    "Chemical Bonding - Covalent": "Covalent bonds form when non-metal atoms share pairs of electrons to achieve a full outer shell. Simple molecular substances have weak intermolecular forces (low melting/boiling points); giant covalent structures like diamond have very high melting points.",
    "Chemical Bonding - Metallic": "Metallic bonding is the electrostatic attraction between a lattice of positive metal ions and a 'sea' of delocalised electrons. This explains malleability, ductility, and electrical/thermal conductivity of metals.",
    "Formulae and Equations": "Chemical formulae represent the ratio of atoms/ions in a compound. Balanced equations conserve atoms of each element on both sides, and state symbols (s, l, g, aq) describe physical states.",
    "The Mole Concept": "One mole of any substance contains Avogadro's number (6.02x10^23) of particles. Moles link mass, molar mass, and number of particles, and for gases at r.t.p. one mole occupies 24 dm^3.",
    "Stoichiometry Calculations": "Stoichiometry uses balanced equations and mole ratios to calculate reacting masses, volumes, and concentrations, including limiting reagent and percentage yield problems.",
    "Acids and Bases": "Acids release H+ ions in water and turn litmus red; bases/alkalis release OH- ions and turn litmus blue. pH measures H+ concentration; neutralisation is acid + base forming salt and water.",
    "Salts and Preparation of Salts": "Salts are prepared by reacting an acid with a metal, base, carbonate, or via titration/precipitation, chosen based on the solubility of the salt and reactant.",
    "Qualitative Analysis (Tests for Ions and Gases)": "Standard tests identify cations (flame tests, NaOH/NH3 precipitates), anions (e.g. dilute acid for carbonate, AgNO3 for halides), and gases (litmus for ammonia, glowing splint for oxygen, limewater for CO2).",
    "Redox Reactions": "Oxidation is loss of electrons (or gain of oxygen); reduction is gain of electrons (or loss of oxygen). Oxidising agents gain electrons; reducing agents lose electrons — tracked via oxidation states.",
    "Electrolysis": "Electrolysis uses electrical energy to decompose an ionic compound in molten or aqueous form. Cations move to the cathode and are reduced; anions move to the anode and are oxidised; product depends on ion discharge preference.",
    "Energy Changes": "Exothermic reactions release heat to the surroundings (temperature rises); endothermic reactions absorb heat (temperature falls). Energy profile diagrams show activation energy and overall enthalpy change.",
    "Rate of Reaction": "Reaction rate increases with higher concentration, temperature, surface area, or a catalyst, because these increase the frequency and/or energy of successful particle collisions.",
    "Reversible Reactions and Equilibrium": "Reversible reactions reach dynamic equilibrium when forward and reverse rates are equal. Le Chatelier's principle predicts how changing concentration, pressure, or temperature shifts the position of equilibrium.",
    "Periodic Table Trends": "Elements are arranged by increasing proton number into periods and groups; elements in the same group share similar outer-shell electron configurations and so similar chemical properties.",
    "Metals and the Reactivity Series": "The reactivity series ranks metals by how readily they lose electrons, predicting displacement reactions and reactions with water, acid, and oxygen.",
    "Extraction of Metals": "The extraction method depends on a metal's position in the reactivity series: electrolysis for very reactive metals, reduction with carbon for moderately reactive metals, and native occurrence for unreactive metals.",
    "Chemistry of Air and the Environment": "Clean air is a mixture of gases (mainly nitrogen and oxygen); pollutants like CO, SO2, NOx, and particulates arise from combustion and contribute to acid rain, smog, and the greenhouse effect.",
    "Chemistry of Water": "Water treatment involves sedimentation, filtration, and chlorination to make water potable; tests for water use anhydrous copper(II) sulfate (turns blue) and cobalt(II) chloride paper (turns pink).",
    "Organic Chemistry - Alkanes and Alkenes": "Alkanes are saturated hydrocarbons (single bonds only); alkenes are unsaturated (contain a C=C double bond) and react with bromine water, decolourising it, unlike alkanes.",
    "Organic Chemistry - Alcohols and Carboxylic Acids": "Alcohols (e.g. ethanol) can be oxidised to carboxylic acids (e.g. ethanoic acid); carboxylic acids react with alcohols to form esters, and behave as weak acids with carbonates and metals.",
    "Polymers": "Polymers are large molecules made of repeating monomer units. Addition polymers form from unsaturated monomers (e.g. alkenes) joining without loss of atoms; condensation polymers form with loss of a small molecule like water.",
    "Experimental Design and Techniques": "Good experimental design controls variables, uses appropriate apparatus (e.g. titration, filtration, distillation) for the separation/measurement needed, and considers sources of error and precision.",
}

FLASHCARDS = {
    "Kinetic Particle Theory": [("What happens to particle spacing and energy on heating a solid?", "Particles vibrate more and gain kinetic energy; spacing increases until the lattice breaks down (melting).")],
    "Atomic Structure": [("How do isotopes of the same element differ?", "They have the same proton number but different neutron numbers (different nucleon number).")],
    "Chemical Bonding - Ionic": [("Why do ionic compounds conduct electricity only when molten or aqueous?", "Ions are fixed in a rigid lattice in the solid state; melting or dissolving frees them to move and carry charge.")],
    "Chemical Bonding - Covalent": [("Why does diamond have a very high melting point?", "It's a giant covalent structure — every atom is bonded to 4 others by strong covalent bonds that require huge energy to break.")],
    "Chemical Bonding - Metallic": [("Why are metals malleable?", "Layers of metal ions can slide over each other while the delocalised electron sea keeps the structure held together.")],
    "Formulae and Equations": [("Why must equations be balanced?", "To conserve atoms of each element — mass cannot be created or destroyed in a chemical reaction.")],
    "The Mole Concept": [("How many particles are in 1 mole of a substance?", "6.02 x 10^23 (Avogadro's number).")],
    "Stoichiometry Calculations": [("How do you find the limiting reagent?", "Compare mole ratios of reactants available to the balanced equation's required ratio; the one that runs out first is limiting.")],
    "Acids and Bases": [("What ion do all acids release in water?", "H+ (hydrogen ions).")],
    "Salts and Preparation of Salts": [("How do you prepare an insoluble salt?", "By precipitation — mixing two soluble solutions whose ions combine to form the insoluble salt.")],
    "Qualitative Analysis (Tests for Ions and Gases)": [("How do you test for carbon dioxide gas?", "Bubble through limewater — it turns milky/cloudy.")],
    "Redox Reactions": [("What happens to the oxidising agent in a redox reaction?", "It gains electrons (is reduced).")],
    "Electrolysis": [("Which electrode do cations move to and what happens there?", "The cathode; they are reduced (gain electrons).")],
    "Energy Changes": [("Is bond breaking exothermic or endothermic?", "Endothermic — energy must be absorbed to break bonds.")],
    "Rate of Reaction": [("Why does increasing temperature increase reaction rate?", "Particles move faster, collide more often, and more collisions exceed the activation energy.")],
    "Reversible Reactions and Equilibrium": [("What does Le Chatelier's principle predict?", "The system shifts to oppose an imposed change, e.g. increasing pressure shifts equilibrium toward fewer gas molecules.")],
    "Periodic Table Trends": [("Why do elements in the same group have similar properties?", "They have the same number of electrons in their outer shell.")],
    "Metals and the Reactivity Series": [("What does the reactivity series predict?", "Which metals can displace others from compounds, and how vigorously a metal reacts with water/acid/oxygen.")],
    "Extraction of Metals": [("How is a very reactive metal like aluminium extracted?", "By electrolysis, since it's too reactive to be reduced by carbon.")],
    "Chemistry of Air and the Environment": [("What causes acid rain?", "Sulfur dioxide and nitrogen oxides dissolving in atmospheric water to form acidic solutions.")],
    "Chemistry of Water": [("How do you test for the presence of water?", "Anhydrous copper(II) sulfate turns from white to blue.")],
    "Organic Chemistry - Alkanes and Alkenes": [("How can you distinguish an alkene from an alkane?", "Add bromine water — alkenes decolourise it, alkanes do not.")],
    "Organic Chemistry - Alcohols and Carboxylic Acids": [("What forms when a carboxylic acid reacts with an alcohol?", "An ester (plus water), via esterification.")],
    "Polymers": [("What's the difference between addition and condensation polymerisation?", "Addition polymers form with no loss of atoms; condensation polymers form with loss of a small molecule like water.")],
    "Experimental Design and Techniques": [("Why is a control variable important in an experiment?", "It isolates the effect of the independent variable so the result can be attributed to it alone.")],
}


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_concepts():
    with open(DATA_DIR / "concepts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept"])
        for c in CONCEPTS:
            w.writerow([c])


def write_notes():
    with open(DATA_DIR / "notes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "chunk_text"])
        for concept, text in NOTES.items():
            w.writerow([concept, text])


def write_flashcards():
    with open(DATA_DIR / "flashcards.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "concept", "question", "answer"])
        for concept, cards in FLASHCARDS.items():
            for question, answer in cards:
                w.writerow([uuid.uuid4().hex[:8], concept, question, answer])


def simulate_interactions():
    """Simulate ~5 weeks of study history via a per-concept BKT-style generative process."""
    db.init_db()
    # clear existing interactions for a clean demo run
    with db.get_conn() as conn:
        conn.execute("DELETE FROM interactions")
        conn.commit()

    now = datetime.now(timezone.utc)
    start = now - timedelta(weeks=5)

    rows = []
    for concept, tier in CONCEPTS.items():
        p_l0, p_t = DIFFICULTY_PARAMS[tier]
        known = random.random() < p_l0
        n_attempts = random.randint(*ATTEMPTS_RANGE[tier])
        for i in range(n_attempts):
            if known:
                correct = random.random() > P_SLIP
            else:
                correct = random.random() < P_GUESS
            # transition after the opportunity
            if not known and random.random() < p_t:
                known = True

            ts = start + (now - start) * (i + random.random()) / n_attempts
            time_taken = round(random.uniform(15, 45) if tier == "easy" else random.uniform(30, 90), 1)
            is_paper = random.random() < 0.15
            question_id = f"PYP-{random.randint(2018, 2023)}-Q{random.randint(1, 8)}" if is_paper else f"synthetic-{uuid.uuid4().hex[:6]}"
            rows.append((ts, concept, question_id, correct, time_taken))

    rows.sort(key=lambda r: r[0])
    for ts, concept, qid, correct, time_taken in rows:
        db.log_interaction(
            timestamp=ts.isoformat(),
            concept=concept,
            question_id=qid,
            correct=correct,
            time_taken_seconds=time_taken,
            photo_path=None,
        )
    return len(rows)


if __name__ == "__main__":
    _ensure_data_dir()
    write_concepts()
    write_notes()
    write_flashcards()
    n = simulate_interactions()
    print(f"Seeded {len(CONCEPTS)} concepts, {sum(len(v) for v in FLASHCARDS.values())} flashcards, "
          f"{len(NOTES)} note chunks, {n} synthetic interactions.")
