#!/usr/bin/env python3
"""
build_gene_evidence.py -- turn the curation TSV into a 0-5 evidence table.

Why this exists
---------------
The deck currently bins the 82 candidates into Strong / Moderate / Weak /
None. That scale cannot express the three things that actually decide whether
a gene is worth a wet-lab slot:

  * NTNG1 and NPTX2 are CAUSAL -- germline KO and KO+rescue respectively --
    but neither endpoint is pain (lanceolate-ending development; scratching).
    "Strong" is wrong and "Weak" is wrong.
  * KCND3 and DAGLA have perturbation-plus-behaviour claims with NO SOURCE
    retained. The evidence, if it exists, is Tier 4; the confidence is zero.
  * ASIC1's citations point at ASIC3 in proprioceptors and spinal ASIC1a.
    The biology is real and the row is unusable as written.

Folding "how good is the experiment", "which species", "do we know the
direction" and "did anyone read the paper" into one adjective destroys all
four. So: one ORDINAL tier for experimental strength, and three ORTHOGONAL
flags kept as their own columns.

    Tier 0  never tested for pain
    Tier 1  marker / expression only, no perturbation
    Tier 2  correlative change in a pain or injury model
    Tier 3  mechanism shown, but no in vivo pain endpoint
    Tier 4  causal in rodent: perturbation + behaviour
    Tier 5  Tier 4 plus human corroboration

    species           rodent | human | both | n/a
    direction         yes / no / contested / non-pain endpoint
    source            verified | partial | unsourced | mismatched | withdrawn

Tier sets the ceiling. The flags set the discount. A Tier 4 row with
source=unsourced is a LEAD, not a result, and must not be presented as one.

Writes
------
    gene_evidence.csv   one row per gene, the table above
    gene_evidence.md    the slide-20 roster, grouped by tier

Edit the TABLE below as you verify sources; everything downstream reads the CSV.
"""
from __future__ import annotations

import csv
from pathlib import Path

# gene, network_tier, evidence_tier, species, direction, source, note
TABLE = [
    # ---- Tier 5: causal rodent + human corroboration -------------------
    ("PLXND1", "B", 5, "both", "yes (pro-nociceptive)", "verified",
     "Autoantigen of anti-small-DRG-neuron antibodies; 15.8% of idiopathic SFN; patient IgG passive transfer induces neuropathic pain. Normal physiological role still unknown."),

    # ---- Tier 4: perturbation + behaviour in rodent ---------------------
    ("TMEM45B", "B", 4, "rodent", "yes (pro-nociceptive)", "verified",
     "Whole-body KO + intrathecal siRNA; selective loss of inflammatory/incisional mechanical hypersensitivity. NOT a conditional KO. Human function untested."),
    ("FXYD6", "A", 4, "rodent", "yes (pro-nociceptive)", "verified",
     "SNS-Cre conditional KO, impaired noxious heat and capsaicin responses, reduced capsaicin currents; binds TRPV1. Strongest row in the set."),
    ("ALKAL2", "B", 4, "rodent", "yes (pro-nociceptive)", "verified",
     "DRG depletion and ALK blockade (crizotinib, lorlatinib) reverse thermal hyperalgesia and mechanical allodynia. Clinical-stage tool compounds already exist."),
    ("GFRA1", "B", 4, "rodent", "contested (bidirectional)", "partial",
     "GDNF overexpression analgesic after nerve injury; peripheral GDNF causes muscle hyperalgesia via IB4+/GFRa1+ neurons. Direction is context-dependent, not a property of the gene."),
    ("ASIC1", "B", 4, "both", "contested (site-dependent)", "mismatched",
     "Established acid-sensing family. Row citations point at ASIC3 in proprioceptors and spinal ASIC1a, not DRG ASIC1. Central ASIC1a KO can INCREASE pain. Re-cite isoform- and site-specific papers before use."),
    ("NTRK3", "B", 4, "rodent", "yes (pro-nociceptive, paclitaxel only)", "withdrawn",
     "proNT3 -> TrkC -> CCL2 in chemotherapy-induced neuropathic pain. The original EMBO Rep 2024 paper was WITHDRAWN for mislabelling proNT3 as NT3; cite the 2025 replacement. Primary identity is proprioceptor, not nociceptor."),
    ("NPTX2", "B", 4, "rodent", "non-pain endpoint (itch)", "verified",
     "KO + DRG-specific rescue + dominant negative, all on repetitive scratching via spinal GRPR+ neurons. The 'no effect on pain' claim is not in the abstract -- cite the figure or drop it."),
    ("NTNG1", "B", 4, "rodent", "non-pain endpoint (development)", "verified",
     "Germline KO; lanceolate-ending elaboration and Meissner/Pacinian corpuscle development. Structural, not nociceptive. Human orthologue role not validated."),
    ("KCND3", "B", 4, "rodent", "yes (anti-nociceptive) - unverified", "unsourced",
     "Reported knockdown -> mechanical hypersensitivity, downregulated in neuropathic models, C-LTMR tactile pain. NO SOURCE RETAINED. The wider Kv4/KChIP/DPP10 complex also controls mechanical pain, so specificity needs care."),
    ("DAGLA", "B", 4, "rodent", "contested", "unsourced",
     "DAGLa-KO reported hypoalgesic; peripheral DAGL inhibition reported analgesic; acetaminophen partly DAGL/CB1-dependent. NO SOURCE RETAINED and DAGLb has the clearer phenotype."),

    # ---- Tier 3: mechanism, no in vivo pain endpoint --------------------
    ("CHRNA3", "B", 3, "rodent", "no", "verified",
     "Marks mechanoinsensitive 'silent' nociceptors; becomes mechanosensitive via NGF/PIEZO2. Evidence rests on a Chrna3-EGFP BAC REPORTER, not endogenous protein. Literature says PEPTIDERGIC; both assigned layers are C-NP -- investigate, do not score as confirmation."),
    ("SYT2", "B", 3, "both", "no", "partial",
     "Fast Ca2+ sensor at central inhibitory synapses; human LoF causes presynaptic congenital myasthenic syndrome. No DRG study at all. Confirms the generic-hub flag -- the one row where deprioritising for genericness is justified."),
    ("GPM6A", "A*", 3, "rodent", "no", "unsourced",
     "Neurite elongation in vitro; detected in DRG membrane proteomes from neuropathic models. NO SOURCE RETAINED."),
    ("DAPK1", "B", 3, "rodent", "no", "unsourced",
     "Co-localises with ATF3 and NMDA-NR2B in DRG. Descriptive only. NO SOURCE RETAINED and the source may be thesis-level -- check quality before citing."),

    # ---- Tier 2: correlative change in a pain / injury model ------------
    ("SCN3B", "A*", 2, "both", "no", "verified",
     "Nav beta3 is the dominant auxiliary subunit in small rat DRG neurons; mRNA up in CCI; modulates Nav1.8 in oocytes. mRNA only, no KO pain phenotype. Would be missed by a 'SCN3B' keyword search -- the key paper only ever says 'beta3'."),
    ("SLC7A14", "A", 2, "rodent", "no", "partial",
     "Upregulated as a proteomic DEP in a disease model and in rat ACC after SNI. Post hoc hit among many DEPs; needs orthogonal validation."),
    ("VSNL1", "B", 2, "rodent", "no", "unsourced",
     "VILIP-1 reported consistently DOWN across rodent neuropathic DRG transcriptomic datasets. Biomarker-like signature, no causal study. NO SOURCE RETAINED."),
    ("PLXNA2", "B", 2, "rodent", "no", "unsourced",
     "Reported upregulated in DRG in both CCI and CFA. NO SOURCE RETAINED, no KO study."),
    ("UGT8", "B", 2, "rodent", "no", "unsourced",
     "Reported down in DRG SATELLITE GLIA in painful diabetic neuropathy. WRONG COMPARTMENT for a neuronal-subtype assignment -- move to a glial track or drop. NO SOURCE RETAINED."),
    ("LITAF", "A*", 2, "human", "opposite phenotype", "unsourced",
     "LITAF/SIMPLE mutations cause CMT1C: demyelinating neuropathy with sensory LOSS. Strong human genetics, wrong direction. Do not count as pain-target support."),

    # ---- Tier 1: marker / expression only -------------------------------
    ("SCGN", "B", 1, "both", "no", "partial",
     "~7% of mouse DRG neurons, 98% CGRP+ and none IB4+; dorsal-horn fibres conserved to human; rhizotomy reduces staining. One of the few rows where literature and layer assignment agree on peptidergic identity."),
    ("CACNG5", "B", 1, "human", "no", "underspecified",
     "TARP gamma-5 names the human C-PEP.TAC1/CACNG5 subtype. Validates the atlas, not the target -- circular by construction. Name the atlas release."),
    ("GPRC5C", "A", 1, "rodent", "no", "partial",
     "Ciliary transition zone of olfactory sensory neurons; mRNA in TG bulk RNA-seq; paralog GPRC5B down in spinal dorsal horn after nerve injury. The Wangzhou citation compares native vs CULTURED DRG, not pain vs control -- removed."),
    ("RAMP3", "A*", 1, "both", "no", "not citable",
     "Partners CLR to form the adrenomedullin AM2 receptor and a weak CGRP receptor. RAMP1, not RAMP3, is the migraine target. Replace the encyclopaedia topic page with primary pharmacology."),
    ("CREB3L1", "A*", 1, "rodent", "no", "unsourced",
     "No CREB3L1-specific pain data. Paralog Luman/CREB3 is an axonally synthesised retrograde regeneration signal. ER-stress biology, not nociception."),
    ("SDC3", "B", 1, "rodent", "no", "unsourced",
     "Paralog SDC1 is induced in injured CGRP/IB4 nociceptors. SDC3 itself unstudied. Dominant layer is the ATF3 injury state, not a resting subtype."),
    ("SERPINA1", "B", 1, "n/a", "no", "none",
     "Very high fold change (log2FC 5.76 in C-LTMR.CDH9) but alpha-1-antitrypsin is a high-abundance plasma protein -- check for blood/serum contamination before treating it as neuronal signal."),
    ("CD9", "B", 1, "rodent", "no", "none",
     "Tetraspanin at peripheral-nerve and spinal-root paranodes; important for paranode formation. Axon/myelin organisation, not nociception."),
    ("HS6ST2", "B", 1, "rodent", "no", "none",
     "Heparan-sulfate 6-O-sulfotransferase with an axon-guidance role. Developmental only."),
    ("ISL2", "B", 1, "rodent", "no", "none",
     "Islet-2 governs DRG sensory differentiation. Developmental only."),

    # ---- Tier 0: never tested for pain ---------------------------------
    ("HPCA", "B", 0, "rodent", "paralog tested, negative", "partial",
     "No HPCA-specific data. Paralog HPCAL4 makes only a LIMITED contribution to pain and itch. One of the few rows where family evidence argues AGAINST the gene."),
    ("AHNAK", "B", 0, "rodent", "no", "none",
     "Known pro-inflammatory mast-cell activity, no neuronal data. If the atlas puts AHNAK in a neuronal cluster, check for ambient-RNA / immune-cell contamination."),
    ("MYO1B", "B", 0, "n/a", "no", "none",
     "Mechanotransduction role is an inference from the protein family, not a finding. Keep clearly separated from evidence-backed rows."),
    ("DDR2", "B", 0, "n/a", "no", "none",
     "Collagen-sensing RTK, arthritis/fibrosis context. Joint-pain link is plausible -- re-search explicitly before closing out."),
    ("CEMIP", "B", 0, "n/a", "no", "none",
     "KIAA1199 hyaluronidase, osteoarthritis synovium context. Worth a targeted joint-pain re-search."),
    ("RBP4", "B", 0, "n/a", "no", "none",
     "Retinol-binding protein 4 adipokine with diabetes associations -- re-search against diabetic neuropathy explicitly."),
    ("EMP1", "B", 0, "n/a", "no", "none",
     "PMP22-family tetraspan protein. Family membership makes a myelin/neuropathy angle worth a targeted re-search."),
    ("COL25A1", "B", 0, "n/a", "no", "none",
     "Brain membrane collagen (CLAC-P). Very high fold change (log2FC 5.82 in C-LTMR.CDH9) but no functional data."),
]

#: Rows with no retained claim beyond "no direct DRG/nociception functional
#: literature found in targeted searches" plus a one-line protein description.
#: ABSENCE OF HITS IS NOT EVIDENCE OF ABSENCE -- every one of these must be
#: re-searched with protein aliases and rodent ortholog casing before the
#: verdict is final. SCN3B is the standing warning: its key paper only ever
#: says "beta3".
UNTESTED = {
    "A*": {
        "PKN1": "Protein kinase N1; CNS/ALS neurofilament role reported",
        "SLC10A6": "Sodium-dependent sulfate/bile-acid transporter",
        "HSD17B13": "Hepatic lipid-droplet dehydrogenase",
        "LNX1": "E3 ligase / PDZ scaffold",
        "CDH1": "E-cadherin epithelial adhesion molecule",
        "FHL2": "LIM-domain scaffold (heart/liver/bone)",
    },
    "B": {
        "TECR": "Fatty-acid elongation enzyme",
        "MPPED1": "Orphan metallophosphoesterase",
        "LY6H": "Ly6 nicotinic-AChR modulator (CNS)",
        "TRARG1": "Insulin-responsive trafficking protein",
        "SERINC2": "Serine incorporator",
        "BAIAP2L1": "I-BAR cytoskeletal protein",
        "C11ORF87": "Uncharacterized neuronal membrane protein",
        "SLC2A12": "GLUT12 glucose transporter",
        "BCL11B": "CTIP2 transcription factor (skin/CNS identity)",
        "CPNE5": "Copine-5 Ca2+-dependent membrane protein",
        "SPINT1": "HAI-1 protease inhibitor (epithelial)",
        "NSG1": "NEEP21 endosomal trafficking of glutamate receptors",
        "FSTL4": "Follistatin-like 4",
        "HSPB6": "Small heat-shock protein Hsp20 (muscle/vascular)",
        "LDB2": "LIM-domain binding 2 (developmental)",
        "NELL1": "Osteogenic EGF-like protein",
        "UST": "Uronyl-2-sulfotransferase (GAG); plausible ECM link",
        "AMOTL1": "Angiomotin-like / Hippo pathway",
        "IGSF3": "Ig-superfamily adhesion molecule",
        "IGFBP6": "IGF-II binding protein",
        "BTN2A2": "Butyrophilin immune co-inhibitor",
        "ARHGAP18": "Rho GAP; cytoskeleton",
        "BRINP2": "BMP/retinoic-acid-inducible neural protein",
        "PTPDC1": "Phosphatase domain-containing 1",
        "SLC5A3": "Myo-inositol transporter (osmoregulation)",
        "PLK2": "Polo-like kinase 2/SNK; synaptic scaling in CNS",
        "TPD52L1": "Tumor-protein D52-like",
        "RTP5": "Receptor-transporting protein 5",
        "SDK2": "Sidekick-2 synaptic adhesion",
        "NECTIN1": "Adhesion molecule / HSV receptor",
        "SERTM1": "Serine-rich transmembrane module 1; orphan",
        "NPNT": "Nephronectin ECM/integrin ligand",
        "TSPAN18": "Tetraspanin-18",
        "ID4": "Inhibitor-of-differentiation bHLH protein",
        "MDGA1": "Inhibitory-synapse regulator",
        "ASPHD2": "Aspartate beta-hydroxylase domain 2; orphan",
        "AMIGO2": "Adhesion molecule / neurite outgrowth",
    },
}

TIER_NAME = {
    0: "Never tested for pain",
    1: "Marker or expression only",
    2: "Correlative change in a pain model",
    3: "Mechanism, no in vivo pain endpoint",
    4: "Causal in rodent: perturbation + behaviour",
    5: "Causal in rodent + human corroboration",
}

FIELDS = ["gene", "network_tier", "evidence_tier", "evidence_tier_name",
          "species", "direction_established", "source_status", "note"]


def rows():
    out = [dict(zip(["gene", "network_tier", "evidence_tier", "species",
                     "direction_established", "source_status", "note"], r))
           for r in TABLE]
    for net, genes in UNTESTED.items():
        for g, desc in genes.items():
            out.append(dict(gene=g, network_tier=net, evidence_tier=0, species="n/a",
                            direction_established="no", source_status="none",
                            note=f"{desc}. No direct DRG/nociception functional literature found. "
                                 f"Re-run the search with protein aliases and rodent ortholog casing."))
    for r in out:
        r["evidence_tier_name"] = TIER_NAME[r["evidence_tier"]]
    seen = {}
    for r in out:
        if r["gene"] in seen:
            raise SystemExit(f"duplicate gene: {r['gene']}")
        seen[r["gene"]] = r
    return sorted(out, key=lambda r: (-r["evidence_tier"], r["gene"]))


def main():
    here = Path(__file__).parent
    data = rows()
    print(f"{len(data)} genes")
    counts = {}
    for r in data:
        counts[r["evidence_tier"]] = counts.get(r["evidence_tier"], 0) + 1
    for t in sorted(counts, reverse=True):
        print(f"  Tier {t}  {TIER_NAME[t]:<45} {counts[t]:>3}")

    with open(here / "gene_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(data)
    print(f"  wrote {here / 'gene_evidence.csv'}")

    lines = ["# Candidate genes by pain-evidence tier", ""]
    for t in sorted(counts, reverse=True):
        sel = [r for r in data if r["evidence_tier"] == t]
        lines += [f"## Tier {t} - {TIER_NAME[t]}  (n={len(sel)})", ""]
        if t == 0:
            lines += ["  " + ", ".join(r["gene"] for r in sel), "",
                      "  Absence of hits is not evidence of absence.", ""]
            continue
        for r in sel:
            flags = []
            if r["source_status"] not in ("verified",):
                flags.append(f"source: {r['source_status']}")
            if "non-pain" in r["direction_established"] or "contested" in r["direction_established"] \
                    or "opposite" in r["direction_established"]:
                flags.append(f"direction: {r['direction_established']}")
            if r["species"] == "rodent":
                flags.append("rodent only")
            tag = ("  [" + "; ".join(flags) + "]") if flags else ""
            lines += [f"- **{r['gene']}** (network tier {r['network_tier']}){tag}",
                      f"  {r['note']}"]
        lines.append("")
    (here / "gene_evidence.md").write_text("\n".join(lines))
    print(f"  wrote {here / 'gene_evidence.md'}")


if __name__ == "__main__":
    main()