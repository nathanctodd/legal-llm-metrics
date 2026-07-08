# Translation QC Notes — subset_100 ES/EU

Source: `subset_100_en.json` (cais/mmlu professional_law test, seed 42, first 100).
Translations preserve `id` and `answer` exactly; `question` + all 4 `choices` translated.
Validated at merge: ids 0-99 complete, 4 choices per item, answer indices identical to EN.

## Source typos cleaned in translation (EN kept as-is)
The English source has typos that translators fixed rather than reproduced, so ES/EU
are marginally cleaner than EN on these items. If one of these flips in the
cross-lingual eval, check here first.
- id 3 choice C: duplicated phrase in EN
- id 5 choice B: "quitclaim deed. quitclaim deed." duplicated in EN
- id 6 choice D: "defendant's plaintiff's conduct" → translated as plaintiff's (doctrinally correct)
- id 16 choice D: "respbnsible"
- id 58 choice C: duplicated "as incorporated into" phrase in EN
- id 87 choice D (EU): stray trailing "f" dropped
- id 92: EN switches orchard→farm mid-question and has a date anomaly; preserved faithfully in both languages

## Terminology decisions
- US common-law terms with no civil-law equivalent are translated descriptively with
  the English in parentheses on first use per item (future interests, quitclaim deed,
  accessory before the fact, judgment as a matter of law, puffing, etc.). On these
  items the answer distinction partly rides on the parenthesized English.
- Harmonized at merge: manslaughter = "homicidio culposo (manslaughter)" (ES),
  "gizahilketa (manslaughter)" (EU); murder = asesinato / erailketa.
- Glossary anchors: plaintiff = demandante / demandatzailea; defendant = demandado
  (civil), acusado / akusatua (criminal); battery = agresión / kontaktu bidezko erasoa;
  assault = amenaza de agresión / mehatxuzko erasoa; negligence = negligencia /
  zabarkeria; burglary = robo con allanamiento (kept distinct from robbery = robo).
- ES register: neutral formal Spanish; "homicidio culposo" chosen over Spain's
  "homicidio imprudente".
- EU register: euskara batua, IVAP-style legal Basque; some property-law terms are
  coined descriptive renderings, not established terminology.
