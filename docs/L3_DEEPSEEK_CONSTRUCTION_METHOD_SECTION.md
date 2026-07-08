# L3 DeepSeek Generation and Cleaning Method Paragraphs

## Recommended Insertion Position

It is recommended to place this part in “3.3.1 Patient-level Attribution Memory”, immediately after the definition of the L3 structured record, and split it into the following three subsections.

## 3.3.1.1 DeepSeek-based Constrained Candidate Attribution Generation

For each ICU stay (s_i) in the training set and each observed medication (m_j), we construct a stay–medication query instance:

[
q_{ij}=\left(h_i,m_j,c_j,\mathcal D_i,\mathcal P_i,\mathcal L_i\right),
]

where (h_i) denotes the hospital identifier, (c_j) denotes the medication function category, and
(\mathcal D_i), (\mathcal P_i), and (\mathcal L_i) denote the diagnoses, procedures, and abnormal laboratory signals observed in this stay, respectively.

We call the DeepSeek API to perform constrained candidate attribution for each (q_{ij}). The task of the model is not to generate new clinical indications, but to select at most one most representative single anchor from the existing context of the current stay:

[
a_{ij}\in
\left{
\operatorname{diag}(d):d\in\mathcal D_i
\right}
\cup
\left{
\operatorname{proc}(p):p\in\mathcal P_i
\right}
\cup
\left{
\operatorname{lab}(l):l\in\mathcal L_i
\right}.
]

If the existing context is insufficient to support reliable attribution, the model must output an empty anchor rather than guessing external indications. The compact output is restricted to structured JSON, containing the stay ID, hospital ID, standardized medication name, medication function category, candidate anchor, and a conservative attribution hypothesis:

[
\tilde e_{ij}^{(L3)}=
\left(
s_i,h_i,m_j,c_j,a_{ij},r_{ij}
\right).
]

Here, (r_{ij}) only describes an empirical attribution indicating that “the medication may be related to the patient context”, and does not express this association as a confirmed prescription intention, therapeutic indication, or causal relationship. For records that cannot be attributed, the system keeps an empty anchor and marks it as `unattributed`, in order to avoid forcibly creating pseudo-explanations.

In the current generation results, a total of 210,263 records contain one candidate anchor, while empty-anchor records are not output; all non-empty records contain a single anchor. Among the non-empty anchors, diagnosis, procedure, and laboratory anchors account for 130,844, 33,960, and 45,459 records, respectively.

## 3.3.1.2 Anchor Validity Verification and Structured Cleaning

After the DeepSeek output is completed, we reconnect the compact L3 records with the original stay metadata according to the stay ID. For each record, the cleaning module sequentially performs the following checks:

1. **Identifier consistency check**: Check whether the stay ID and hospital ID can be located in the original metadata.
2. **Medication consistency check**: Check whether the output medication exists in the observed medication list of the stay.
3. **Anchor type normalization**: Normalize `dx/diag`, `px/proc/treatment`, and `lab` into `diagnosis`, `procedure`, and `lab`, respectively.
4. **Anchor context verification**: Check whether the normalized anchor type–text pair appears in the diagnosis, procedure, or laboratory set of the stay.
5. **Structural schema validation**: Check whether the complete L3 strictly contains predefined fields such as patient, hospital, medication, anchor, related features, attribution type, hypothesis, and statistical data.
6. **Conservative language check**: Detect phrases that may imply definite causality or prescription intention, such as `confirmed indication`, `caused by`, `definitely treated`, and `treatment of`.

For records that pass the verification, the system constructs the complete L3 representation:

[
e_{ij}^{(L3)}=
\left(
s_i,h_i,m_j,c_j,a_{ij},F_i,t_{ij},r_{ij},S_{ij}
\right),
]

where (F_i) contains the diagnoses, procedures, laboratory abnormalities, co-medications, and basic demographic information of the stay; (t_{ij}) denotes the attribution evidence level; and (S_{ij}) denotes the global and current-hospital evidence required for subsequent statistical aggregation.

## 3.3.1.3 Statistical Evidence Completion and Attribution Stratification

After metadata reconnection is completed, we compute anchor–drug co-occurrence evidence based only on historical data. For anchor (a), medication (m), and hospital (h), we calculate:

[
n(a),\quad n(a,m),\quad
p(m\mid a)=\frac{n(a,m)}{n(a)},
]

as well as within-hospital statistics:

[
n_h(a),\quad n_h(a,m),\quad
p_h(m\mid a)=\frac{n_h(a,m)}{n_h(a)}.
]

At the same time, we record the number of hospitals covered by the anchor and the number of hospitals covered by the anchor–drug pair. According to whether a patient-level anchor exists and the strength of global/hospital support, L3 records are divided into:

* `patient_supported_low_statistical_support`: a patient-context anchor exists, but global support is limited;
* `patient_supported_with_hospital_amplification`: the within-hospital conditional frequency is at least 0.10 higher than the global frequency, and the within-hospital pair support is no less than 3;
* `patient_supported`: the global anchor–drug support is no less than 5;
* `patient_supported_weak_evidence`: patient-context support exists, but the above statistical conditions are not yet satisfied.

These labels indicate the strength of empirical support rather than clinical causal levels. The complete L3 is then used only as an intermediate statistical source for constructing L1 consensus memory, L2 hospital residual memory, and weak auxiliary evidence; during online recommendation, the original patient-level L3 records are not directly retrieved.

## Concise Version for Direct Replacement in the Main Text

> **DeepSeek-assisted patient-context attribution.** For each stay–medication observation pair in the training set, we serialize the hospital, medication, medication category, and the diagnosis, procedure, and abnormal laboratory signal sequences of the stay into a structured input, and call DeepSeek to select at most one candidate context anchor. Candidate anchors are only allowed to come from a single diagnosis, procedure, or laboratory abnormality within the current stay; if there is insufficient evidence, the model returns an empty anchor. The output is parsed as structured JSON and reconnected with the original stay metadata to verify the consistency of the stay, hospital, medication, anchor type, and anchor text. Anchors that fail context verification are set to empty or excluded from the anchored aggregation set. Subsequently, the system completes the diagnoses, procedures, laboratory abnormalities, co-medications, and basic demographic features, and computes global and within-hospital anchor–drug support and conditional frequencies only on the training set. The L3 attribution hypothesis only represents an auditable candidate empirical attribution, and does not indicate confirmed prescription intention or clinical causality. Online inference does not directly retrieve the original L3 records; L3 is only used for offline aggregation into L1 consensus memory, L2 hospital residual memory, and attenuated weak auxiliary evidence.
