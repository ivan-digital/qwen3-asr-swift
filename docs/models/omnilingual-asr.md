# Omnilingual ASR — CTC-300M Model

Meta's [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr)
is a language-agnostic speech recognition family covering **1600+ languages**.
This module ports the CTC-300M v2 variant to CoreML / Neural Engine on Apple
Silicon.

- HuggingFace: [aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s](https://huggingface.co/aufklarer/Omnilingual-ASR-CTC-300M-CoreML-INT8-10s)
- Module: `Sources/OmnilingualASR/`
- Weights: ~312 MB (INT8 k-means palettization)
- Parameters: ~326M (wav2vec2 encoder + CTC head)
- Sample rate: 16 kHz
- Window: 10 s fixed (also published: 5 s variant)

## Architecture

Omnilingual CTC is a supervised fine-tune of Meta's SSL wav2vec 2.0 backbone
with a linear CTC head over a shared 10288-entry SentencePiece vocabulary.

```
raw audio [1, samples]
  → wav2vec2 feature extractor (7 CNN layers, ×320 downsample)
  → weight-normalised Conv1d positional encoder
  → 24 × pre-norm Transformer encoder layers (dim=1024, heads=16, ffn=4096)
  → final layer norm
  → linear CTC head → logits [1, T, 10288]
```

`T ≈ samples / 320` — a 10 s window at 16 kHz produces 500 frames of logits.

### Why CTC

- Parallel, non-autoregressive — one forward pass per window, no decoder loop
- Language-agnostic: the 10288-vocab SentencePiece shares across all 1600+
  languages, so no per-language conditioning is needed. The LLM variants
  (a separate follow-up module) are the ones that accept language codes like
  `eng_Latn`; CTC explicitly ignores the `language` hint.
- Streaming-friendly in principle (sliding-window encoder + incremental CTC
  decoder), though the published CoreML/MLX exports are batch-only. Streaming
  is tracked as a follow-up to [#195](https://github.com/soniqo/speech-swift/issues/195).

### Input preprocessing

The CoreML graph is traced at a fixed window — 5 s (80 000 samples) or 10 s
(160 000 samples). Before inference:

1. Resample to 16 kHz mono Float32 (`AudioFileLoader.resample`)
2. Pad or chunk to exactly `config.inputSamples`
3. **Layer-norm waveform normalization**: `(x - mean(x)) / sqrt(var(x) + ε)`.
   Matches fairseq2's `apply_audio_normalization(waveform) = layer_norm(waveform, waveform.shape)`
   used by Meta's upstream `ASRInferencePipeline`. Biased variance (divide by N),
   `ε = 1e-5` matching PyTorch's `layer_norm` default.

### CTC decoding

Greedy argmax, matching `omnilingual_asr/models/inference/pipeline.py`:

```python
pred_ids = torch.argmax(logits, dim=-1)         # [B, T]
mask[1:] = pred_ids[1:] != pred_ids[:-1]        # collapse consecutive duplicates
decoded = pred_ids[mask]
text = tokenizer.decode(decoded, skip_special_tokens=True)
```

There is intentionally **no explicit blank-token filtering** at the decoder
level. Omnilingual's CTC blank id is tied to the SentencePiece `<pad>` token
(id = 1), and is stripped at detokenize time via `skip_special_tokens=True`.

## Config

Runtime config lives in `OmnilingualConfig` and decodes directly from the
published `config.json` on HuggingFace:

| Field | 10s variant | 5s variant |
|---|---|---|
| `sampleRate` | 16000 | 16000 |
| `frameRate` | 50 | 50 |
| `maxAudioSeconds` | 10.0 | 5.0 |
| `inputSamples` | 160 000 | 80 000 |
| `encoder.numLayers` | 24 | 24 |
| `encoder.modelDim` | 1024 | 1024 |
| `encoder.numHeads` | 16 | 16 |
| `ctcHead.vocabSize` | 10288 | 10288 |

## Model variants

The [aufklarer/coreml-speech-models](https://huggingface.co/collections/aufklarer/coreml-speech-models)
and [aufklarer/mlx-speech-models](https://huggingface.co/collections/aufklarer/mlx-speech-models)
collections currently publish:

| Repo | Size | Notes |
|---|---|---|
| `Omnilingual-ASR-CTC-300M-CoreML-INT8` | 312 MB | 5 s fixed window, INT8 k-means palettized, iOS 17+/ANE |
| `Omnilingual-ASR-CTC-300M-CoreML-INT8-10s` | 312 MB | 10 s variant for longer utterances (this module's default) |
| `Omnilingual-ASR-CTC-300M-MLX-4bit` | 193 MB | MLX backend, 326M params |
| `Omnilingual-ASR-CTC-300M-MLX-8bit` | 342 MB | MLX backend, 326M params |
| `Omnilingual-ASR-CTC-1B-MLX-4bit` | 549 MB | MLX backend, 1.01B params |
| `Omnilingual-ASR-CTC-1B-MLX-8bit` | 1006 MB | MLX backend, 1.01B params |
| `Omnilingual-ASR-CTC-3B-MLX-4bit` | 1709 MB | MLX backend, ~3B params |
| `Omnilingual-ASR-CTC-3B-MLX-8bit` | 3159 MB | MLX backend, ~3B params |
| `Omnilingual-ASR-CTC-7B-MLX-4bit` | 3.55 GB | MLX backend, ~7B params (largest CTC variant) |
| `Omnilingual-ASR-CTC-7B-MLX-8bit` | 6.63 GB | MLX backend, ~7B params (largest CTC variant) |

**All 10 published variants are wired through this module:**

- CoreML (5 s / 10 s) via `OmnilingualASRModel.fromPretrained(...)` — `OmnilingualASRModel.shortWindowModelId` and `OmnilingualASRModel.defaultModelId`
- MLX (300M / 1B / 3B / 7B in 4-bit and 8-bit) via `OmnilingualASRMLXModel.fromPretrained(variant: .b1, bits: 4, ...)` — auto-detects variant and bits from the HF model id

Both backends share the `Configuration`, `SentencePieceVocabulary`, `CTCGreedyDecoder`, and waveform `layer_normalize` preprocessing, so behaviour matches modulo runtime differences (CoreML on ANE vs MLX on Metal).

## See also

- [omnilingual-asr-inference.md](../inference/omnilingual-asr-inference.md) —
  runtime pipeline, chunking, VAD integration
- [Omnilingual ASR paper](https://arxiv.org/abs/2511.09690)
- Meta's reference pipeline: [facebookresearch/omnilingual-asr/pipeline.py](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/inference/pipeline.py)


## Supported Languages (full list)

Omnilingual ASR covers **1,672 languages** across 32 distinct scripts. The CTC head is fully language-agnostic — no language hint is needed at inference time. Each entry lists the ISO 639-3 + ISO 15924 code and the English language name (resolved via the CLDR / SIL ISO 639-3 registry). Country hints are shown only for languages with an ISO 639-1 alpha-2 code, where CLDR's primary-region heuristic is reliable; for low-resource languages without alpha-2 codes the country is implied by the language name itself (e.g. "Saint Lucian Creole French", "Egyptian Arabic", "Aja (Benin)").

Source of canonical list: [`lang_ids.py`](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py).

<details>
<summary><b>Click to expand all 1,672 language codes</b> (grouped by script, with names and country hints)</summary>

### Latin (Latn) — 1398 languages

- `aae_Latn` — Arbëreshë Albanian
- `aal_Latn` — Afade
- `abb_Latn` — Bankon
- `abi_Latn` — Abidji
- `abn_Latn` — Abua
- `abp_Latn` — Abellen Ayta
- `abr_Latn` — Abron
- `abs_Latn` — Ambonese Malay
- `aca_Latn` — Achagua
- `acd_Latn` — Gikyode
- `ace_Latn` — Acehnese
- `acf_Latn` — Saint Lucian Creole French
- `ach_Latn` — Acoli
- `acn_Latn` — Achang
- `acr_Latn` — Achi
- `acu_Latn` — Achuar-Shiwiar
- `ade_Latn` — Adele
- `adh_Latn` — Adhola
- `adj_Latn` — Adioukrou
- `aeu_Latn` — Akeu
- `afo_Latn` — Eloyi
- `afr_Latn` — Afrikaans — South Africa
- `agd_Latn` — Agarabi
- `agg_Latn` — Angor
- `agn_Latn` — Agutaynen
- `agr_Latn` — Aguaruna
- `agu_Latn` — Aguacateco
- `aha_Latn` — Ahanta
- `ahk_Latn` — Akha
- `ahl_Latn` — Igo
- `ahs_Latn` — Ashe
- `aia_Latn` — Arosi
- `ajg_Latn` — Aja (Benin)
- `aka_Latn` — Akan — Ghana
- `akb_Latn` — Batak Angkola
- `ake_Latn` — Akawaio
- `akp_Latn` — Siwu
- `ala_Latn` — Alago
- `alj_Latn` — Alangan
- `aln_Latn` — Gheg Albanian
- `alo_Latn` — Larike-Wakasihu
- `alp_Latn` — Alune
- `als_Latn` — Tosk Albanian
- `alz_Latn` — Alur
- `ame_Latn` — Yanesha'
- `amf_Latn` — Hamer-Banna
- `ami_Latn` — Amis
- `amk_Latn` — Ambai
- `amu_Latn` — Guerrero Amuzgo
- `anc_Latn` — Ngas
- `ank_Latn` — Goemai
- `ann_Latn` — Obolo
- `anw_Latn` — Anaang
- `any_Latn` — Anyin
- `aom_Latn` — Ömie
- `aoz_Latn` — Uab Meto
- `apb_Latn` — Sa'a
- `apr_Latn` — Arop-Lokep
- `arg_Latn` — Aragonese — Spain
- `arl_Latn` — Arabela
- `asa_Latn` — Asu
- `asg_Latn` — Cishingini
- `ast_Latn` — Asturian
- `ata_Latn` — Pele-Ata
- `atb_Latn` — Zaiwa
- `atg_Latn` — Ivbie North-Okpela-Arhe
- `ati_Latn` — Attié
- `atq_Latn` — Aralle-Tabulahan
- `avn_Latn` — Avatime
- `avu_Latn` — Avokaya
- `awb_Latn` — Awa (Papua New Guinea)
- `awo_Latn` — Awak
- `ayo_Latn` — Ayoreo
- `ayr_Latn` — Central Aymara
- `ayz_Latn` — Mai Brat
- `aze_Latn` — Azerbaijani — Azerbaijan
- `azg_Latn` — San Pedro Amuzgos Amuzgo
- `azz_Latn` — Highland Puebla Nahuatl
- `bag_Latn` — Tuki
- `bam_Latn` — Bambara — Mali
- `ban_Latn` — Balinese
- `bao_Latn` — Waimaha
- `bas_Latn` — Basaa
- `bav_Latn` — Vengo
- `bax_Latn` — Bamun
- `bba_Latn` — Baatonum
- `bbb_Latn` — Barai
- `bbc_Latn` — Batak Toba
- `bbj_Latn` — Ghomala
- `bbo_Latn` — Northern Bobo Madaré
- `bbu_Latn` — Kulung (Nigeria)
- `bcc_Latn` — Southern Balochi
- `bce_Latn` — Bamenyam
- `bci_Latn` — Baoulé
- `bcl_Latn` — Central Bikol
- `bcs_Latn` — Kohumono
- `bcw_Latn` — Bana
- `bcy_Latn` — Bacama
- `bcz_Latn` — Bainouk-Gunyaamolo
- `bda_Latn` — Bayot
- `bde_Latn` — Bade
- `bdg_Latn` — Bonggi
- `bdh_Latn` — Baka (South Sudan)
- `bdm_Latn` — Buduma
- `bdq_Latn` — Bahnar
- `bdu_Latn` — Oroko
- `beb_Latn` — Bebele
- `beh_Latn` — Biali
- `bem_Latn` — Bemba
- `bep_Latn` — Besoa
- `bew_Latn` — Betawi
- `bex_Latn` — Jur Modo
- `bfa_Latn` — Bari
- `bfd_Latn` — Bafut
- `bfo_Latn` — Malba Birifor
- `bgr_Latn` — Bawm Chin
- `bgt_Latn` — Bughotu
- `bhp_Latn` — Bima
- `bhz_Latn` — Bada (Indonesia)
- `bib_Latn` — Bissa
- `bim_Latn` — Bimoba
- `bis_Latn` — Bislama — Vanuatu
- `biv_Latn` — Southern Birifor
- `bjk_Latn` — Barok
- `bjn_Latn` — Banjar
- `bjr_Latn` — Binumarien
- `bjt_Latn` — Balanta-Ganja
- `bjv_Latn` — Bedjond
- `bjw_Latn` — Bakwé
- `bjz_Latn` — Baruga
- `bkd_Latn` — Binukid
- `bkh_Latn` — Bakoko
- `bkm_Latn` — Kom
- `bkv_Latn` — Bekwarra
- `bky_Latn` — Bokyi
- `ble_Latn` — Balanta-Kentohe
- `blh_Latn` — Kuwaa
- `blt_Latn` — Tai Dam
- `blx_Latn` — Mag-Indi Ayta
- `blz_Latn` — Balantak
- `bmm_Latn` — Northern Betsimisaraka Malagasy
- `bmq_Latn` — Bomu
- `bmr_Latn` — Muinane
- `bmu_Latn` — Somba-Siawari
- `bmv_Latn` — Bum
- `bnm_Latn` — Batanga
- `bnn_Latn` — Bunun
- `bno_Latn` — Bantoanon
- `bnp_Latn` — Bola
- `boa_Latn` — Bora
- `boj_Latn` — Anjam
- `bom_Latn` — Berom
- `bor_Latn` — Borôro
- `bos_Latn` — Bosnian — Bosnia and Herzegovina
- `bou_Latn` — Bondei
- `bov_Latn` — Tuwuli
- `box_Latn` — Buamu
- `bpr_Latn` — Koronadal Blaan
- `bps_Latn` — Sarangani Blaan
- `bqc_Latn` — Boko (Benin)
- `bqg_Latn` — Bago-Kusuntu
- `bqj_Latn` — Bandial
- `bqp_Latn` — Busa
- `bre_Latn` — Breton — France
- `bri_Latn` — Mokpwe
- `bru_Latn` — Eastern Bru
- `bsc_Latn` — Bassari
- `bsj_Latn` — Bangwinji
- `bsk_Latn` — Burushaski
- `bsq_Latn` — Bassa
- `bss_Latn` — Akoose
- `bsy_Latn` — Sabah Bisaya
- `btd_Latn` — Batak Dairi
- `btm_Latn` — Batak Mandailing
- `bts_Latn` — Batak Simalungun
- `btt_Latn` — Bete-Bendi
- `btx_Latn` — Batak Karo
- `bud_Latn` — Ntcham
- `bug_Latn` — Buginese
- `bum_Latn` — Bulu
- `buo_Latn` — Terei
- `bus_Latn` — Bokobaru
- `bux_Latn` — Boghom
- `bvb_Latn` — Bube
- `bvc_Latn` — Baelelea
- `bvz_Latn` — Bauzi
- `bwq_Latn` — Southern Bobo Madaré
- `bwr_Latn` — Bura-Pabir
- `bwu_Latn` — Buli (Ghana)
- `bxf_Latn` — Bilur
- `bxk_Latn` — Bukusu
- `byc_Latn` — Ubaghara
- `byr_Latn` — Baruya
- `bys_Latn` — Burak
- `byv_Latn` — Medumba
- `byx_Latn` — Qaqet
- `bzh_Latn` — Mapos Buang
- `bzj_Latn` — Belize Kriol English
- `bzw_Latn` — Basa (Nigeria)
- `caa_Latn` — Chortí
- `cab_Latn` — Garifuna
- `cac_Latn` — Chuj
- `cak_Latn` — Kaqchikel
- `cap_Latn` — Chipaya
- `car_Latn` — Carib
- `cas_Latn` — Tsimané
- `cat_Latn` — Catalan — Spain
- `cax_Latn` — Chiquitano
- `cbc_Latn` — Carapana
- `cbi_Latn` — Chachi
- `cbr_Latn` — Cashibo-Cacataibo
- `cbs_Latn` — Cashinahua
- `cbt_Latn` — Chayahuita
- `cbu_Latn` — Candoshi-Shapra
- `cbv_Latn` — Cacua
- `cce_Latn` — Chopi
- `ccg_Latn` — Samba Daka
- `cco_Latn` — Comaltepec Chinantec
- `ceb_Latn` — Cebuano
- `ceg_Latn` — Chamacoco
- `cek_Latn` — Eastern Khumi Chin
- `cen_Latn` — Cen
- `ces_Latn` — Czech — Czechia
- `cfa_Latn` — Dijim-Bwilim
- `cfm_Latn` — Falam Chin
- `cgc_Latn` — Kagayanen
- `cgg_Latn` — Chiga
- `chf_Latn` — Tabasco Chontal
- `chq_Latn` — Quiotepec Chinantec
- `chz_Latn` — Ozumacín Chinantec
- `cjk_Latn` — Chokwe
- `cjo_Latn` — Ashéninka Pajonal
- `cjp_Latn` — Cabécar
- `ckl_Latn` — Cibak
- `cko_Latn` — Anufo
- `ckr_Latn` — Kairak
- `cky_Latn` — Cakfem-Mushere
- `cla_Latn` — Ron
- `cle_Latn` — Lealao Chinantec
- `cly_Latn` — Eastern Highland Chatino
- `cme_Latn` — Cerma
- `cmo_Latn` — Central Mnong
- `cmr_Latn` — Mro-Khimi Chin
- `cnh_Latn` — Hakha Chin
- `cni_Latn` — Asháninka
- `cnl_Latn` — Lalana Chinantec
- `cnt_Latn` — Tepetotutla Chinantec
- `coe_Latn` — Koreguaje
- `cof_Latn` — Colorado
- `cok_Latn` — Santa Teresa Cora
- `con_Latn` — Cofán
- `cor_Latn` — Cornish — United Kingdom
- `cot_Latn` — Caquinte
- `cou_Latn` — Wamey
- `cpa_Latn` — Palantla Chinantec
- `cpb_Latn` — Ucayali-Yurúa Ashéninka
- `cpu_Latn` — Pichis Ashéninka
- `cpy_Latn` — South Ucayali Ashéninka
- `crk_Latn` — Plains Cree
- `crn_Latn` — El Nayar Cora
- `crq_Latn` — Iyo'wujwa Chorote
- `crs_Latn` — Seselwa Creole French
- `crt_Latn` — Iyojwa'ja Chorote
- `csk_Latn` — Jola-Kasa
- `cso_Latn` — Sochiapam Chinantec
- `ctd_Latn` — Tedim Chin
- `cte_Latn` — Tepinapa Chinantec
- `ctl_Latn` — Tlacoatzintepec Chinantec
- `cto_Latn` — Emberá-Catío
- `ctu_Latn` — Chol
- `cuc_Latn` — Usila Chinantec
- `cui_Latn` — Cuiba
- `cuk_Latn` — San Blas Kuna
- `cul_Latn` — Culina
- `cut_Latn` — Teutila Cuicatec
- `cux_Latn` — Tepeuxila Cuicatec
- `cwa_Latn` — Kabwa
- `cwe_Latn` — Kwere
- `cwt_Latn` — Kuwaataay
- `cya_Latn` — Nopala Chatino
- `cym_Latn` — Welsh — United Kingdom
- `daa_Latn` — Dangaléat
- `dag_Latn` — Dagbani
- `dah_Latn` — Gwahatike
- `dan_Latn` — Danish — Denmark
- `dav_Latn` — Taita
- `dbd_Latn` — Dadiya
- `dbj_Latn` — Ida'an
- `dbq_Latn` — Daba
- `ddn_Latn` — Dendi (Benin)
- `ded_Latn` — Dedua
- `deg_Latn` — Degema
- `des_Latn` — Desano
- `deu_Latn` — German — Germany
- `dga_Latn` — Southern Dagaare
- `dgh_Latn` — Dghwede
- `dgi_Latn` — Northern Dagara
- `dgk_Latn` — Dagba
- `dgr_Latn` — Dogrib
- `did_Latn` — Didinga
- `dig_Latn` — Digo
- `dik_Latn` — Southwestern Dinka
- `dip_Latn` — Northeastern Dinka
- `dje_Latn` — Zarma
- `djk_Latn` — Eastern Maroon Creole
- `dnj_Latn` — Dan
- `dnt_Latn` — Mid Grand Valley Dani
- `dnw_Latn` — Western Dani
- `dop_Latn` — Lukpa
- `dos_Latn` — Dogosé
- `dru_Latn` — Rukai
- `dsb_Latn` — Lower Sorbian
- `dsh_Latn` — Daasanach
- `dtp_Latn` — Central Dusun
- `dts_Latn` — Toro So Dogon
- `dua_Latn` — Duala
- `dug_Latn` — Duruma
- `dwr_Latn` — Dawro
- `dyi_Latn` — Djimini Senoufo
- `dyo_Latn` — Jola-Fonyi
- `dyu_Latn` — Dyula
- `dzg_Latn` — Dazaga
- `ebu_Latn` — Embu
- `ego_Latn` — Eggon
- `eip_Latn` — Eipomek
- `eiv_Latn` — Askopan
- `eka_Latn` — Ekajuk
- `ekk_Latn` — Standard Estonian
- `eko_Latn` — Koti
- `ekr_Latn` — Yace
- `elm_Latn` — Eleme
- `emp_Latn` — Northern Emberá
- `enb_Latn` — Markweeta
- `eng_Latn` — English — United States
- `enx_Latn` — Enxet
- `epo_Latn` — Esperanto
- `ese_Latn` — Ese Ejja
- `ess_Latn` — Central Siberian Yupik
- `esu_Latn` — Central Yupik
- `eto_Latn` — Eton (Cameroon)
- `ets_Latn` — Yekhee
- `etu_Latn` — Ejagham
- `eus_Latn` — Basque — Spain
- `ewe_Latn` — Ewe — Ghana
- `ewo_Latn` — Ewondo
- `eyo_Latn` — Keiyo
- `eza_Latn` — Ezaa
- `fal_Latn` — South Fali
- `fan_Latn` — Fang
- `fao_Latn` — Faroese — Faroe Islands
- `far_Latn` — Fataleka
- `fat_Latn` — Fanti
- `fia_Latn` — Nobiin
- `fij_Latn` — Fijian — Fiji
- `fil_Latn` — Filipino
- `fin_Latn` — Finnish — Finland
- `fip_Latn` — Fipa
- `fkk_Latn` — Kirya-Konzəl
- `flr_Latn` — Fuliiru
- `fmp_Latn` — Fe'fe'
- `fon_Latn` — Fon
- `fra_Latn` — French — France
- `frd_Latn` — Fordata
- `fry_Latn` — Western Frisian — Netherlands
- `fub_Latn` — Adamawa Fulfulde
- `fuc_Latn` — Pulaar
- `fue_Latn` — Borgu Fulfulde
- `ful_Latn` — Fula — Senegal
- `fuq_Latn` — Central-Eastern Niger Fulfulde
- `fuv_Latn` — Nigerian Fulfulde
- `gag_Latn` — Gagauz
- `gai_Latn` — Borei
- `gam_Latn` — Kandawo
- `gbi_Latn` — Galela
- `gbo_Latn` — Northern Grebo
- `gbr_Latn` — Gbagyi
- `gby_Latn` — Gbari
- `gcc_Latn` — Mali
- `gde_Latn` — Gude
- `gdf_Latn` — Guduf-Gava
- `geb_Latn` — Kire
- `gej_Latn` — Gen
- `ges_Latn` — Geser-Gorom
- `gid_Latn` — Gidar
- `gil_Latn` — Gilbertese
- `giz_Latn` — South Giziga
- `gjn_Latn` — Gonja
- `gkn_Latn` — Gokana
- `gle_Latn` — Irish — Ireland
- `glg_Latn` — Galician — Spain
- `glv_Latn` — Manx — Isle of Man
- `glw_Latn` — Glavda
- `gmv_Latn` — Gamo
- `gna_Latn` — Kaansa
- `gnd_Latn` — Zulgo-Gemzek
- `gng_Latn` — Ngangam
- `gof_Latn` — Gofa
- `gog_Latn` — Gogo
- `gol_Latn` — Gola
- `gor_Latn` — Gorontalo
- `gqr_Latn` — Gor
- `gri_Latn` — Ghari
- `grn_Latn` — Guarani — Paraguay
- `gsl_Latn` — Gusilay
- `gso_Latn` — Southwest Gbaya
- `gub_Latn` — Guajajára
- `guc_Latn` — Wayuu
- `gud_Latn` — Yocoboué Dida
- `gug_Latn` — Paraguayan Guaraní
- `guh_Latn` — Guahibo
- `gui_Latn` — Eastern Bolivian Guaraní
- `gum_Latn` — Guambiano
- `guo_Latn` — Guayabero
- `guq_Latn` — Aché
- `gur_Latn` — Frafra
- `guu_Latn` — Yanomamö
- `gux_Latn` — Gourmanchéma
- `guz_Latn` — Gusii
- `gvc_Latn` — Guanano
- `gvl_Latn` — Gulay
- `gwe_Latn` — Gweno
- `gwi_Latn` — Gwichʼin
- `gwr_Latn` — Gwere
- `gym_Latn` — Ngäbere
- `gyr_Latn` — Guarayu
- `gyz_Latn` — Geji
- `had_Latn` — Hatam
- `hag_Latn` — Hanga
- `hah_Latn` — Hahon
- `hak_Latn` — Hakka Chinese
- `hao_Latn` — Hakö
- `hap_Latn` — Hupla
- `hat_Latn` — Haitian Creole — Haiti
- `hau_Latn` — Hausa — Nigeria
- `haw_Latn` — Hawaiian
- `hay_Latn` — Haya
- `hbb_Latn` — Huba
- `hch_Latn` — Huichol
- `heh_Latn` — Hehe
- `her_Latn` — Herero — Namibia
- `hia_Latn` — Lamang
- `hif_Latn` — Fiji Hindi
- `hig_Latn` — Kamwe
- `hil_Latn` — Hiligaynon
- `hkk_Latn` — Hunjara-Kaina Ke
- `hla_Latn` — Halia
- `hlt_Latn` — Matu Chin
- `hnn_Latn` — Hanunoo
- `hns_Latn` — Caribbean Hindustani
- `hrv_Latn` — Croatian — Croatia
- `hsb_Latn` — Upper Sorbian
- `hto_Latn` — Minica Huitoto
- `hub_Latn` — Huambisa
- `hue_Latn` — San Francisco Del Mar Huave
- `hui_Latn` — Huli
- `hul_Latn` — Hula
- `hun_Latn` — Hungarian — Hungary
- `hus_Latn` — Huastec
- `huu_Latn` — Murui Huitoto
- `huv_Latn` — San Mateo Del Mar Huave
- `hux_Latn` — Nüpode Huitoto
- `hvn_Latn` — Sabu
- `hwc_Latn` — Hawai'i Creole English
- `hwo_Latn` — Hwana
- `iba_Latn` — Iban
- `ibb_Latn` — Ibibio
- `ibo_Latn` — Igbo — Nigeria
- `icr_Latn` — Islander Creole English
- `ida_Latn` — Idakho-Isukha-Tiriki
- `idd_Latn` — Ede Idaca
- `idu_Latn` — Idoma
- `ifa_Latn` — Amganad Ifugao
- `ifb_Latn` — Batad Ifugao
- `ife_Latn` — Ifè
- `ifk_Latn` — Tuwali Ifugao
- `ifu_Latn` — Mayoyao Ifugao
- `ify_Latn` — Keley-I Kallahan
- `igl_Latn` — Igala
- `ign_Latn` — Ignaciano
- `ijc_Latn` — Izon
- `ijn_Latn` — Kalabari
- `ikk_Latn` — Ika
- `ikw_Latn` — Ikwere
- `ilb_Latn` — Ila
- `ilo_Latn` — Iloko
- `imo_Latn` — Imbongu
- `ina_Latn` — Interlingua
- `inb_Latn` — Inga
- `ind_Latn` — Indonesian — Indonesia
- `iou_Latn` — Tuma-Irumu
- `ipi_Latn` — Ipili
- `ipk_Latn` — Inupiaq — United States
- `iqw_Latn` — Ikwo
- `iri_Latn` — Rigwe
- `irk_Latn` — Iraqw
- `ish_Latn` — Esan
- `isl_Latn` — Icelandic — Iceland
- `iso_Latn` — Isoko
- `ita_Latn` — Italian — Italy
- `its_Latn` — Isekiri
- `itv_Latn` — Itawit
- `itw_Latn` — Ito
- `itz_Latn` — Itzá
- `ixl_Latn` — Ixil
- `izr_Latn` — Izere
- `izz_Latn` — Izii
- `jac_Latn` — Popti'
- `jal_Latn` — Yalahatan
- `jam_Latn` — Jamaican Creole English
- `jav_Latn` — Javanese — Indonesia
- `jax_Latn` — Jambi Malay
- `jbu_Latn` — Jukun Takum
- `jen_Latn` — Dza
- `jic_Latn` — Tol
- `jiv_Latn` — Shuar
- `jmc_Latn` — Machame
- `jmd_Latn` — Yamdena
- `jmx_Latn` — Western Juxtlahuaca Mixtec
- `jqr_Latn` — Jaqaru
- `juk_Latn` — Wapan
- `juo_Latn` — Jiba
- `jvn_Latn` — Caribbean Javanese
- `kab_Latn` — Kabyle
- `kac_Latn` — Kachin
- `kai_Latn` — Karekare
- `kaj_Latn` — Jju
- `kak_Latn` — Kalanguya
- `kam_Latn` — Kamba
- `kao_Latn` — Xaasongaxango
- `kaq_Latn` — Capanahua
- `kay_Latn` — Kamayurá
- `kbl_Latn` — Kanembu
- `kbo_Latn` — Keliko
- `kbp_Latn` — Kabiyè
- `kbq_Latn` — Kamano
- `kbr_Latn` — Kafa
- `kbt_Latn` — Abadi
- `kby_Latn` — Manga Kanuri
- `kcg_Latn` — Tyap
- `kcn_Latn` — Nubi
- `kcq_Latn` — Kamo
- `kdc_Latn` — Kutu
- `kde_Latn` — Makonde
- `kdh_Latn` — Tem
- `kdi_Latn` — Kumam
- `kdj_Latn` — Karamojong
- `kdl_Latn` — Tsikimba
- `kdn_Latn` — Kunda
- `kea_Latn` — Kabuverdianu
- `kek_Latn` — Kekchí
- `ken_Latn` — Kenyang
- `keo_Latn` — Kakwa
- `ker_Latn` — Kera
- `keu_Latn` — Akebu
- `kez_Latn` — Kukele
- `kfw_Latn` — Kharam Naga
- `kha_Latn` — Khasi
- `khq_Latn` — Koyra Chiini
- `kia_Latn` — Kim
- `kij_Latn` — Kilivila
- `kik_Latn` — Kikuyu — Kenya
- `kin_Latn` — Kinyarwanda — Rwanda
- `kix_Latn` — Khiamniungan Naga
- `kjb_Latn` — Q'anjob'al
- `kjc_Latn` — Coastal Konjo
- `kje_Latn` — Kisar
- `kjg_Latn` — Khmu
- `kjk_Latn` — Highland Konjo
- `kki_Latn` — Kagulu
- `kkj_Latn` — Kako
- `kln_Latn` — Kalenjin
- `kls_Latn` — Kalasha
- `klu_Latn` — Klao
- `klv_Latn` — Maskelynes
- `klw_Latn` — Tado
- `kma_Latn` — Konni
- `kmd_Latn` — Majukayang Kalinga
- `kml_Latn` — Tanudan Kalinga
- `kmr_Latn` — Northern Kurdish
- `kmu_Latn` — Kanite
- `kmy_Latn` — Koma
- `kna_Latn` — Dera (Nigeria)
- `knb_Latn` — Lubuagan Kalinga
- `knc_Latn` — Central Kanuri
- `kne_Latn` — Kankanaey
- `knf_Latn` — Mankanya
- `knj_Latn` — Western Kanjobal
- `knk_Latn` — Kuranko
- `kno_Latn` — Kono (Sierra Leone)
- `kog_Latn` — Cogui
- `kol_Latn` — Kol (Papua New Guinea)
- `koo_Latn` — Konzo
- `kpo_Latn` — Ikposo
- `kpq_Latn` — Korupun-Sela
- `kps_Latn` — Tehit
- `kpz_Latn` — Kupsabiny
- `kqe_Latn` — Kalagan
- `kqo_Latn` — Eastern Krahn
- `kqp_Latn` — Kimré
- `kqr_Latn` — Kimaragang
- `kri_Latn` — Krio
- `krj_Latn` — Kinaray-a
- `krl_Latn` — Karelian
- `krs_Latn` — Gbaya (Sudan)
- `krx_Latn` — Karon
- `ksb_Latn` — Shambala
- `ksd_Latn` — Kuanua
- `ksf_Latn` — Bafia
- `ksr_Latn` — Borong
- `kss_Latn` — Southern Kisi
- `ktj_Latn` — Plapo Krumen
- `kto_Latn` — Kuot
- `kua_Latn` — Kuanyama — Namibia
- `kub_Latn` — Kutep
- `kue_Latn` — Kuman (Papua New Guinea)
- `kuh_Latn` — Kushi
- `kus_Latn` — Kusaal
- `kvn_Latn` — Border Kuna
- `kvw_Latn` — Wersing
- `kwd_Latn` — Kwaio
- `kwf_Latn` — Kwara'ae
- `kwi_Latn` — Awa-Cuaiquer
- `kwm_Latn` — Kwambi
- `kxf_Latn` — Manumanaw Karen
- `kyb_Latn` — Butbut Kalinga
- `kyc_Latn` — Kyaka
- `kyf_Latn` — Kouya
- `kyg_Latn` — Keyagana
- `kyo_Latn` — Kelon
- `kyq_Latn` — Kenga
- `kyx_Latn` — Rapoisi
- `kyz_Latn` — Kayabí
- `kzf_Latn` — Da'a Kaili
- `kzi_Latn` — Kelabit
- `lac_Latn` — Lacandon
- `lag_Latn` — Langi
- `laj_Latn` — Lango (Uganda)
- `lam_Latn` — Lamba
- `las_Latn` — Lama (Togo)
- `lat_Latn` — Latin — Holy See (Vatican City State)
- `lav_Latn` — Latvian — Latvia
- `law_Latn` — Lauje
- `lbw_Latn` — Tolaki
- `lcm_Latn` — Tungag
- `ldb_Latn` — Dũya
- `led_Latn` — Lendu
- `lee_Latn` — Lyélé
- `lef_Latn` — Lelemi
- `lem_Latn` — Nomaande
- `lew_Latn` — Ledo Kaili
- `lex_Latn` — Luang
- `lgg_Latn` — Lugbara
- `lgl_Latn` — Wala
- `lhu_Latn` — Lahu
- `lia_Latn` — West-Central Limba
- `lid_Latn` — Nyindrou
- `lij_Latn` — Ligurian
- `lin_Latn` — Lingala — Congo, The Democratic Republic of the
- `lip_Latn` — Sekpele
- `lir_Latn` — Liberian English
- `lit_Latn` — Lithuanian — Lithuania
- `lje_Latn` — Rampi
- `ljp_Latn` — Lampung Api
- `lkb_Latn` — Kabras
- `lke_Latn` — Kenyi
- `lla_Latn` — Lala-Roba
- `lld_Latn_gherd` — Ladin — gherd dialect
- `lld_Latn_valbadia` — Ladin — valbadia dialect
- `llg_Latn` — Lole
- `lln_Latn` — Lele (Chad)
- `lme_Latn` — Pévé
- `lnd_Latn` — Lundayeh
- `lns_Latn` — Lamnso'
- `lnu_Latn` — Longuda
- `loa_Latn` — Loloda
- `lob_Latn` — Lobi
- `lok_Latn` — Loko
- `lom_Latn` — Loma (Liberia)
- `lon_Latn` — Malawi Lomwe
- `loq_Latn` — Lobala
- `lsi_Latn` — Lashi
- `lsm_Latn` — Saamia
- `ltg_Latn` — Latgalian
- `lth_Latn` — Thur
- `lto_Latn` — Tsotso
- `ltz_Latn` — Luxembourgish — Luxembourg
- `lua_Latn` — Luba-Lulua
- `luc_Latn` — Aringa
- `lug_Latn` — Ganda — Uganda
- `luo_Latn` — Luo (Kenya and Tanzania)
- `lus_Latn` — Mizo
- `lwg_Latn` — Wanga
- `lwo_Latn` — Luwo
- `lww_Latn` — Lewo
- `lzz_Latn` — Laz
- `maa_Latn` — San Jerónimo Tecóatl Mazatec
- `mab_Latn` — Yutanduchi Mixtec
- `mad_Latn` — Madurese
- `maf_Latn` — Mafa
- `mah_Latn` — Marshallese — Marshall Islands
- `maj_Latn` — Jalapa De Díaz Mazatec
- `mak_Latn` — Makasar
- `mam_Latn` — Mam
- `maq_Latn` — Chiquihuitlán Mazatec
- `mau_Latn` — Huautla Mazatec
- `maw_Latn` — Mampruli
- `max_Latn` — North Moluccan Malay
- `maz_Latn` — Central Mazahua
- `mbb_Latn` — Western Bukidnon Manobo
- `mbc_Latn` — Macushi
- `mbh_Latn` — Mangseng
- `mbj_Latn` — Nadëb
- `mbt_Latn` — Matigsalug Manobo
- `mbu_Latn` — Mbula-Bwazza
- `mca_Latn` — Maca
- `mcb_Latn` — Machiguenga
- `mcd_Latn` — Sharanahua
- `mcf_Latn` — Matsés
- `mco_Latn` — Coatlán Mixe
- `mcp_Latn` — Makaa
- `mcq_Latn` — Ese
- `mcu_Latn` — Cameroon Mambila
- `mcx_Latn` — Mpiemo
- `mda_Latn` — Mada (Nigeria)
- `mdd_Latn` — Mbum
- `mdv_Latn` — Santa Lucía Monteverde Mixtec
- `med_Latn` — Melpa
- `mee_Latn` — Mengen
- `meh_Latn` — Southwestern Tlaxiaco Mixtec
- `mej_Latn` — Meyah
- `mek_Latn` — Mekeo
- `mel_Latn` — Central Melanau
- `men_Latn` — Mende
- `meq_Latn` — Merey
- `mer_Latn` — Meru
- `met_Latn` — Mato
- `meu_Latn` — Motu
- `mev_Latn` — Mano
- `mfe_Latn` — Morisyen
- `mfh_Latn` — Matal
- `mfi_Latn` — Wandala
- `mfk_Latn` — North Mofu
- `mfm_Latn` — Marghi South
- `mfn_Latn` — Cross River Mbembe
- `mfo_Latn` — Mbe
- `mfq_Latn` — Moba
- `mfv_Latn` — Mandjak
- `mfy_Latn` — Mayo
- `mfz_Latn` — Mabaan
- `mgd_Latn` — Moru
- `mge_Latn` — Mango
- `mgg_Latn` — Mpumpong
- `mgh_Latn` — Makhuwa-Meetto
- `mgi_Latn` — Lijili
- `mgo_Latn` — Metaʼ
- `mhi_Latn` — Ma'di
- `mhk_Latn` — Mungaka
- `mhu_Latn` — Digaro-Mishmi
- `mhx_Latn` — Maru
- `mhy_Latn` — Ma'anyan
- `mib_Latn` — Atatláhuca Mixtec
- `mie_Latn` — Ocotepec Mixtec
- `mif_Latn` — Mofu-Gudur
- `mig_Latn` — San Miguel El Grande Mixtec
- `mih_Latn` — Chayuco Mixtec
- `mil_Latn` — Peñoles Mixtec
- `mim_Latn` — Alacatlatzala Mixtec
- `min_Latn` — Minangkabau
- `mio_Latn` — Pinotepa Nacional Mixtec
- `mip_Latn` — Apasco-Apoala Mixtec
- `miq_Latn` — Mískito
- `mit_Latn` — Southern Puebla Mixtec
- `miu_Latn` — Cacaloxtepec Mixtec
- `miy_Latn` — Ayutla Mixtec
- `miz_Latn` — Coatzospan Mixtec
- `mkf_Latn` — Miya
- `mkl_Latn` — Mokole
- `mkn_Latn` — Kupang Malay
- `mlg_Latn` — Malagasy — Madagascar
- `mlq_Latn` — Western Maninkakan
- `mlt_Latn` — Maltese — Malta
- `mmc_Latn` — Michoacán Mazahua
- `mmg_Latn` — North Ambrym
- `mnb_Latn` — Muna
- `mne_Latn` — Naba
- `mnf_Latn` — Mundani
- `mnk_Latn` — Mandinka
- `mnx_Latn` — Manikion
- `moa_Latn` — Mwan
- `mog_Latn` — Mongondow
- `mop_Latn` — Mopán Maya
- `mor_Latn` — Moro
- `mos_Latn` — Mossi
- `mox_Latn` — Molima
- `moz_Latn` — Mukulu
- `mpg_Latn` — Marba
- `mpm_Latn` — Yosondúa Mixtec
- `mpp_Latn` — Migabac
- `mpx_Latn` — Misima-Panaeati
- `mqb_Latn` — Mbuko
- `mqf_Latn` — Momuna
- `mqj_Latn` — Mamasa
- `mqn_Latn` — Moronene
- `mqy_Latn` — Manggarai
- `mri_Latn` — Māori — New Zealand
- `mrt_Latn` — Marghi Central
- `mrw_Latn` — Maranao
- `msh_Latn` — Masikoro Malagasy
- `msi_Latn` — Sabah Malay
- `msw_Latn` — Mansoanka
- `msy_Latn` — Aruamu
- `mtd_Latn` — Mualang
- `mtj_Latn` — Moskona
- `mto_Latn` — Totontepec Mixe
- `mtu_Latn` — Tututepec Mixtec
- `mtx_Latn` — Tidaá Mixtec
- `mua_Latn` — Mundang
- `mug_Latn` — Musgu
- `muh_Latn` — Mündü
- `mui_Latn` — Musi
- `mur_Latn` — Murle
- `muy_Latn` — Muyang
- `mvp_Latn` — Duri
- `mwq_Latn` — Mün Chin
- `mwv_Latn` — Mentawai
- `mxb_Latn` — Tezoatlán Mixtec
- `mxq_Latn` — Juquila Mixe
- `mxs_Latn` — Huitepec Mixtec
- `mxt_Latn` — Jamiltepec Mixtec
- `mxu_Latn` — Mada (Cameroon)
- `mxv_Latn` — Metlatónoc Mixtec
- `mxy_Latn` — Southeastern Nochixtlán Mixtec
- `myb_Latn` — Mbay
- `myk_Latn` — Mamara Senoufo
- `myx_Latn` — Masaaba
- `myy_Latn` — Macuna
- `mza_Latn` — Santa María Zacatepec Mixtec
- `mzi_Latn` — Ixcatlán Mazatec
- `mzj_Latn` — Manya
- `mzk_Latn` — Nigeria Mambila
- `mzl_Latn` — Mazatlán Mixe
- `mzm_Latn` — Mumuye
- `mzw_Latn` — Deg
- `nab_Latn` — Southern Nambikuára
- `nag_Latn` — Naga Pidgin
- `nal_Latn` — Nalik
- `nan_Latn` — Min Nan Chinese
- `nap_Latn` — Neapolitan
- `nas_Latn` — Naasioi
- `naw_Latn` — Nawuri
- `nbh_Latn` — Ngamo
- `nca_Latn` — Iyo
- `ncf_Latn` — Notsi
- `nch_Latn` — Central Huasteca Nahuatl
- `ncj_Latn` — Northern Puebla Nahuatl
- `ncl_Latn` — Michoacán Nahuatl
- `nco_Latn` — Sibe
- `ncu_Latn` — Chumburung
- `ncx_Latn` — Central Puebla Nahuatl
- `ndi_Latn` — Samba Leko
- `ndj_Latn` — Ndamba
- `ndo_Latn` — Ndonga — Namibia
- `ndp_Latn` — Ndo
- `ndv_Latn` — Ndut
- `ndy_Latn` — Lutos
- `ndz_Latn` — Ndogo
- `neb_Latn` — Toura (Côte d'Ivoire)
- `nfa_Latn` — Dhao
- `nfr_Latn` — Nafaanra
- `nga_Latn` — Ngbaka
- `ngi_Latn` — Ngizim
- `ngl_Latn` — Lomwe
- `ngp_Latn` — Ngulu
- `ngu_Latn` — Guerrero Nahuatl
- `nhe_Latn` — Eastern Huasteca Nahuatl
- `nhg_Latn` — Tetelcingo Nahuatl
- `nhi_Latn` — Zacatlán-Ahuacatlán-Tepetzintla Nahuatl
- `nhn_Latn` — Central Nahuatl
- `nhq_Latn` — Huaxcaleca Nahuatl
- `nhu_Latn` — Noone
- `nhw_Latn` — Western Huasteca Nahuatl
- `nhx_Latn` — Isthmus-Mecayapan Nahuatl
- `nhy_Latn` — Northern Oaxaca Nahuatl
- `nia_Latn` — Nias
- `nij_Latn` — Ngaju
- `nim_Latn` — Nilamba
- `nin_Latn` — Ninzo
- `nja_Latn` — Nzanyi
- `nko_Latn` — Nkonya
- `nla_Latn` — Ngombale
- `nlc_Latn` — Nalca
- `nld_Latn` — Dutch — Netherlands
- `nlg_Latn` — Gela
- `nlk_Latn` — Ninia Yali
- `nlv_Latn` — Orizaba Nahuatl
- `nmg_Latn` — Kwasio
- `nmz_Latn` — Nawdm
- `nnb_Latn` — Nande
- `nnh_Latn` — Ngiemboon
- `nnq_Latn` — Ngindo
- `nnw_Latn` — Southern Nuni
- `noa_Latn` — Woun Meu
- `nob_Latn` — Norwegian Bokmål — Norway
- `not_Latn` — Nomatsiguenga
- `npl_Latn` — Southeastern Puebla Nahuatl
- `npy_Latn` — Napu
- `nso_Latn` — Northern Sotho
- `nst_Latn` — Tase Naga
- `nsu_Latn` — Sierra Negra Nahuatl
- `ntm_Latn` — Nateni
- `ntr_Latn` — Delo
- `nuj_Latn` — Nyole
- `nup_Latn` — Nupe-Nupe-Tako
- `nus_Latn` — Nuer
- `nuz_Latn` — Tlamacazapa Nahuatl
- `nwb_Latn` — Nyabwa
- `nxq_Latn` — Naxi
- `nya_Latn` — Nyanja — Malawi
- `nyf_Latn` — Giryama
- `nyn_Latn` — Nyankole
- `nyo_Latn` — Nyoro
- `nyu_Latn` — Nyungwe
- `nyy_Latn` — Nyakyusa-Ngonde
- `nzi_Latn` — Nzima
- `obo_Latn` — Obo Manobo
- `oci_Latn` — Occitan — France
- `odu_Latn` — Odual
- `ogo_Latn` — Khana
- `ojb_Latn` — Northwestern Ojibwa
- `oku_Latn` — Oku
- `old_Latn` — Mochi
- `omw_Latn` — South Tairora
- `onb_Latn` — Lingao
- `ood_Latn` — Tohono O'odham
- `orc_Latn` — Orma
- `orm_Latn` — Oromo — Ethiopia
- `ote_Latn` — Mezquital Otomi
- `otq_Latn` — Querétaro Otomi
- `ozm_Latn` — Koonzime
- `pab_Latn` — Parecís
- `pad_Latn` — Paumarí
- `pag_Latn` — Pangasinan
- `pam_Latn` — Pampanga
- `pao_Latn` — Northern Paiute
- `pap_Latn` — Papiamento
- `pau_Latn` — Palauan
- `pbb_Latn` — Páez
- `pbc_Latn` — Patamona
- `pbi_Latn` — Parkwa
- `pbs_Latn` — Central Pame
- `pcm_Latn` — Nigerian Pidgin
- `pex_Latn` — Petats
- `pez_Latn` — Eastern Penan
- `pib_Latn` — Yine
- `pil_Latn` — Yom
- `pip_Latn` — Pero
- `pir_Latn` — Piratapuyo
- `pis_Latn` — Pijin
- `piy_Latn` — Piya-Kwonci
- `pjt_Latn` — Pitjantjatjara
- `pkb_Latn` — Pokomo
- `pko_Latn` — Pökoot
- `pls_Latn` — San Marcos Tlacoyalco Popoloca
- `plt_Latn` — Plateau Malagasy
- `plw_Latn` — Brooke's Point Palawano
- `pmf_Latn` — Pamona
- `pmq_Latn` — Northern Pame
- `pms_Latn` — Piedmontese
- `pmy_Latn` — Papuan Malay
- `pne_Latn` — Western Penan
- `pny_Latn` — Pinyin
- `poc_Latn` — Poqomam
- `poe_Latn` — San Juan Atzingo Popoloca
- `poh_Latn` — Poqomchi'
- `poi_Latn` — Highland Popoluca
- `pol_Latn` — Polish — Poland
- `por_Latn` — Portuguese — Brazil
- `pov_Latn` — Upper Guinea Crioulo
- `pow_Latn` — San Felipe Otlaltepec Popoloca
- `poy_Latn` — Pogolo
- `ppk_Latn` — Uma
- `pps_Latn` — San Luís Temalacayuca Popoloca
- `prf_Latn` — Paranan
- `prk_Latn` — Parauk
- `prq_Latn` — Ashéninka Perené
- `pse_Latn` — Central Malay
- `pss_Latn` — Kaulong
- `ptu_Latn` — Bambam
- `pua_Latn` — Western Highland Purepecha
- `pui_Latn` — Puinave
- `pwg_Latn` — Gapapaiwa
- `pwn_Latn` — Paiwan
- `pxm_Latn` — Quetzaltepec Mixe
- `qub_Latn` — Huallaga Huánuco Quechua
- `quc_Latn` — Kʼicheʼ
- `quf_Latn` — Lambayeque Quechua
- `qug_Latn` — Chimborazo Highland Quichua
- `quh_Latn` — South Bolivian Quechua
- `qul_Latn` — North Bolivian Quechua
- `qum_Latn` — Sipacapense
- `qup_Latn` — Southern Pastaza Quechua
- `qur_Latn` — Yanahuanca Pasco Quechua
- `qus_Latn` — Santiago del Estero Quichua
- `quv_Latn` — Sacapulteco
- `quw_Latn` — Tena Lowland Quichua
- `qux_Latn` — Yauyos Quechua
- `quy_Latn` — Ayacucho Quechua
- `quz_Latn` — Cusco Quechua
- `qva_Latn` — Ambo-Pasco Quechua
- `qvc_Latn` — Cajamarca Quechua
- `qve_Latn` — Eastern Apurímac Quechua
- `qvh_Latn` — Huamalíes-Dos de Mayo Huánuco Quechua
- `qvi_Latn` — Imbabura Highland Quichua
- `qvj_Latn` — Loja Highland Quichua
- `qvl_Latn` — Cajatambo North Lima Quechua
- `qvm_Latn` — Margos-Yarowilca-Lauricocha Quechua
- `qvn_Latn` — North Junín Quechua
- `qvo_Latn` — Napo Lowland Quechua
- `qvs_Latn` — San Martín Quechua
- `qvw_Latn` — Huaylla Wanca Quechua
- `qvz_Latn` — Northern Pastaza Quichua
- `qwa_Latn` — Corongo Ancash Quechua
- `qwh_Latn` — Huaylas Ancash Quechua
- `qws_Latn` — Sihuas Ancash Quechua
- `qxa_Latn` — Chiquián Ancash Quechua
- `qxh_Latn` — Panao Huánuco Quechua
- `qxl_Latn` — Salasaca Highland Quichua
- `qxn_Latn` — Northern Conchucos Ancash Quechua
- `qxo_Latn` — Southern Conchucos Ancash Quechua
- `qxp_Latn` — Puno Quechua
- `qxr_Latn` — Cañar Highland Quichua
- `qxt_Latn` — Santa Ana de Tusi Pasco Quechua
- `qxu_Latn` — Arequipa-La Unión Quechua
- `qxw_Latn` — Jauja Wanca Quechua
- `rag_Latn` — Logooli
- `rai_Latn` — Ramoaaina
- `rap_Latn` — Rapanui
- `raw_Latn` — Rawang
- `rej_Latn` — Rejang
- `rel_Latn` — Rendille
- `rgu_Latn` — Ringgou
- `rhg_Latn` — Rohingya
- `rif_Latn` — Riffian
- `rim_Latn` — Nyaturu
- `rmc_Latn` — Carpathian Romani
- `rmo_Latn` — Sinte Romani
- `rmy_Latn` — Vlax Romani
- `rng_Latn` — Ronga
- `rnl_Latn` — Ranglong
- `rob_Latn` — Tae'
- `rof_Latn` — Rombo
- `roh_Latn_surs1244` — Romansh — surs1244 dialect — Switzerland
- `rol_Latn` — Romblomanon
- `ron_Latn` — Romanian — Romania
- `roo_Latn` — Rotokas
- `rop_Latn` — Kriol
- `rro_Latn` — Waima
- `rth_Latn` — Ratahan
- `rub_Latn` — Gungu
- `ruc_Latn` — Ruuli
- `ruf_Latn` — Luguru
- `rug_Latn` — Roviana
- `run_Latn` — Rundi — Burundi
- `rwm_Latn` — Amba (Uganda)
- `sab_Latn` — Buglere
- `sag_Latn` — Sango — Central African Republic
- `saj_Latn` — Sahu
- `saq_Latn` — Samburu
- `sas_Latn` — Sasak
- `sau_Latn` — Saleman
- `say_Latn` — Saya
- `sba_Latn` — Ngambay
- `sbd_Latn` — Southern Samo
- `sbl_Latn` — Botolan Sambal
- `sbp_Latn` — Sangu
- `sch_Latn` — Sakachep
- `scn_Latn` — Sicilian
- `sco_Latn` — Scots
- `sda_Latn` — Toraja-Sa'dan
- `sdo_Latn` — Bukar-Sadung Bidayuh
- `sea_Latn` — Semai
- `seh_Latn` — Sena
- `sei_Latn` — Seri
- `ses_Latn` — Koyraboro Senni
- `sey_Latn` — Secoya
- `sgb_Latn` — Mag-antsi Ayta
- `shi_Latn` — Tachelhit
- `shk_Latn` — Shilluk
- `sho_Latn` — Shanga
- `shp_Latn` — Shipibo-Conibo
- `sid_Latn` — Sidamo
- `sig_Latn` — Paasaal
- `sil_Latn` — Tumulung Sisaala
- `siw_Latn` — Siwai
- `sja_Latn` — Epena
- `sjm_Latn` — Mapun
- `sjr_Latn` — Siar-Lak
- `skg_Latn` — Sakalava Malagasy
- `sld_Latn` — Sissala
- `slk_Latn` — Slovak — Slovakia
- `slu_Latn` — Selaru
- `slv_Latn` — Slovenian — Slovenia
- `sml_Latn` — Central Sama
- `smo_Latn` — Samoan — Samoa
- `sna_Latn` — Shona — Zimbabwe
- `snc_Latn` — Sinaugoro
- `sne_Latn` — Bau Bidayuh
- `snk_Latn` — Soninke
- `snn_Latn` — Siona
- `snp_Latn` — Siane
- `snv_Latn` — Sa'ban
- `snw_Latn` — Selee
- `sol_Latn` — Solos
- `som_Latn` — Somali — Somalia
- `soy_Latn` — Miyobe
- `spa_Latn` — Spanish — Spain
- `spp_Latn` — Supyire Senoufo
- `sps_Latn` — Saposa
- `spy_Latn` — Sabaot
- `src_Latn` — Logudorese Sardinian
- `srd_Latn` — Sardinian — Italy
- `sri_Latn` — Siriano
- `srm_Latn` — Saramaccan
- `srn_Latn` — Sranan Tongo
- `sro_Latn` — Campidanese Sardinian
- `srr_Latn` — Serer
- `ste_Latn` — Liana-Seti
- `stn_Latn` — Owa
- `stp_Latn` — Southeastern Tepehuan
- `sua_Latn` — Sulka
- `suc_Latn` — Western Subanon
- `suk_Latn` — Sukuma
- `sun_Latn` — Sundanese — Indonesia
- `sur_Latn` — Mwaghavul
- `sus_Latn` — Susu
- `suv_Latn` — Puroik
- `swe_Latn` — Swedish — Sweden
- `swh_Latn` — Swahili (individual language)
- `sxb_Latn` — Suba
- `sxn_Latn` — Sangir
- `sya_Latn` — Siang
- `syl_Latn` — Sylheti
- `sza_Latn` — Semelai
- `szy_Latn` — Sakizaya
- `tac_Latn` — Lowland Tarahumara
- `tan_Latn` — Tangale
- `tao_Latn` — Yami
- `tap_Latn` — Taabwa
- `taq_Latn` — Tamasheq
- `tar_Latn` — Central Tarahumara
- `tav_Latn` — Tatuyo
- `tay_Latn` — Atayal
- `tbc_Latn` — Takia
- `tbf_Latn` — Mandara
- `tbg_Latn` — North Tairora
- `tbk_Latn` — Calamian Tagbanwa
- `tbl_Latn` — Tboli
- `tby_Latn` — Tabaru
- `tbz_Latn` — Ditammari
- `tca_Latn` — Ticuna
- `tcc_Latn` — Datooga
- `tcf_Latn` — Malinaltepec Me'phaa
- `tcz_Latn` — Thado Chin
- `tdj_Latn` — Tajio
- `tdn_Latn` — Tondano
- `tdx_Latn` — Tandroy-Mahafaly Malagasy
- `ted_Latn` — Tepo Krumen
- `tee_Latn` — Huehuetla Tepehua
- `tem_Latn` — Timne
- `teo_Latn` — Teso
- `ter_Latn` — Tereno
- `tew_Latn` — Tewa (USA)
- `tex_Latn` — Tennet
- `tfr_Latn` — Teribe
- `tgc_Latn` — Tigak
- `tgj_Latn` — Tagin
- `tgl_Latn` — Filipino — Philippines
- `tgo_Latn` — Sudest
- `tgp_Latn` — Tangoa
- `thk_Latn` — Tharaka
- `tih_Latn` — Timugon Murut
- `tik_Latn` — Tikar
- `tio_Latn` — Teop
- `tkg_Latn` — Tesaka Malagasy
- `tkr_Latn` — Tsakhur
- `tlb_Latn` — Tobelo
- `tli_Latn` — Tlingit
- `tlj_Latn` — Talinga-Bwisi
- `tlp_Latn` — Filomena Mata-Coahuitlán Totonac
- `tly_Latn` — Talysh
- `tmc_Latn` — Tumak
- `tmf_Latn` — Toba-Maskoy
- `tna_Latn` — Tacana
- `tng_Latn` — Tobanga
- `tnk_Latn` — Kwamera
- `tnn_Latn` — North Tanna
- `tnp_Latn` — Whitesands
- `tnr_Latn` — Ménik
- `tnt_Latn` — Tontemboan
- `tob_Latn` — Toba
- `toc_Latn` — Coyutla Totonac
- `toh_Latn` — Gitonga
- `tok_Latn` — Toki Pona
- `tom_Latn` — Tombulu
- `top_Latn` — Papantla Totonac
- `tos_Latn` — Highland Totonac
- `tpi_Latn` — Tok Pisin
- `tpl_Latn` — Tlacoapa Me'phaa
- `tpm_Latn` — Tampulma
- `tpp_Latn` — Pisaflores Tepehua
- `tpt_Latn` — Tlachichilco Tepehua
- `tpz_Latn` — Tinputz
- `tqp_Latn` — Tomoip
- `trc_Latn` — Copala Triqui
- `tri_Latn` — Trió
- `trn_Latn` — Trinitario
- `trp_Latn` — Kok Borok
- `trq_Latn` — San Martín Itunyoso Triqui
- `trs_Latn` — Chicahuaxtla Triqui
- `trv_Latn` — Taroko
- `tsn_Latn` — Tswana — South Africa
- `tso_Latn` — Tsonga — South Africa
- `tsz_Latn` — Purepecha
- `ttc_Latn` — Tektiteko
- `tte_Latn` — Bwanabwana
- `ttj_Latn` — Tooro
- `ttr_Latn` — Tera
- `ttu_Latn` — Torau
- `tue_Latn` — Tuyuca
- `tuf_Latn` — Central Tunebo
- `tui_Latn` — Tupuri
- `tuk_Latn` — Turkmen — Turkmenistan
- `tul_Latn` — Tula
- `tuo_Latn` — Tucano
- `tuq_Latn` — Tedaga
- `tur_Latn` — Turkish — Türkiye
- `tuv_Latn` — Turkana
- `tuy_Latn` — Tugen
- `tvo_Latn` — Tidore
- `tvu_Latn` — Tunen
- `tvw_Latn` — Sedoa
- `twb_Latn` — Western Tawbuid
- `twe_Latn` — Tewa (Indonesia)
- `twu_Latn` — Termanu
- `txa_Latn` — Tombonuo
- `txq_Latn` — Tii
- `txs_Latn` — Tonsea
- `txu_Latn` — Kayapó
- `txy_Latn` — Tanosy Malagasy
- `tye_Latn` — Kyanga
- `tzh_Latn` — Tzeltal
- `tzj_Latn` — Tz'utujil
- `tzo_Latn` — Tzotzil
- `ubl_Latn` — Buhi'non Bikol
- `ubu_Latn` — Umbu-Ungu
- `udl_Latn` — Wuzlam
- `udu_Latn` — Uduk
- `ukv_Latn` — Kuku
- `umb_Latn` — Umbundu
- `upv_Latn` — Uripiv-Wala-Rano-Atchin
- `ura_Latn` — Urarina
- `urb_Latn` — Urubú-Kaapor
- `urd_Latn` — Urdu — Pakistan
- `urh_Latn` — Urhobo
- `urt_Latn` — Urat
- `ury_Latn` — Orya
- `usp_Latn` — Uspanteco
- `uzb_Latn` — Uzbek — Uzbekistan
- `uzn_Latn` — Northern Uzbek
- `vag_Latn` — Vagla
- `vai_Latn` — Vai
- `var_Latn` — Huarijio
- `ver_Latn` — Mom Jango
- `vid_Latn` — Vidunda
- `vie_Latn` — Vietnamese — Viet Nam
- `vif_Latn` — Vili
- `vmc_Latn` — Juxtlahuaca Mixtec
- `vmj_Latn` — Ixtayutla Mixtec
- `vmm_Latn` — Mitlatongo Mixtec
- `vmp_Latn` — Soyaltepec Mazatec
- `vmw_Latn` — Makhuwa
- `vmy_Latn` — Ayautla Mazatec
- `vmz_Latn` — Mazatlán Mazatec
- `vro_Latn` — Võro
- `vun_Latn` — Vunjo
- `vut_Latn` — Vute
- `wal_Latn` — Wolaytta
- `wap_Latn` — Wapishana
- `war_Latn` — Waray
- `waw_Latn` — Waiwai
- `way_Latn` — Wayana
- `wba_Latn` — Warao
- `wbl_Latn` — Wakhi
- `wci_Latn` — Waci Gbe
- `weo_Latn` — Wemale
- `wes_Latn` — Cameroon Pidgin
- `wja_Latn` — Waja
- `wji_Latn` — Warji
- `wlo_Latn` — Wolio
- `wlx_Latn` — Wali (Ghana)
- `wmw_Latn` — Mwani
- `wob_Latn` — Wè Northern
- `wof_Latn` — Gambian Wolof
- `wol_Latn` — Wolof — Senegal
- `wwa_Latn` — Waama
- `xdy_Latn` — Malayic Dayak
- `xed_Latn` — Hdi
- `xer_Latn` — Xerénte
- `xho_Latn` — Xhosa — South Africa
- `xkl_Latn` — Mainstream Kenyah
- `xmm_Latn` — Manado Malay
- `xmv_Latn` — Antankarana Malagasy
- `xnj_Latn` — Ngoni (Tanzania)
- `xog_Latn` — Soga
- `xon_Latn` — Konkomba
- `xpe_Latn` — Liberia Kpelle
- `xrb_Latn` — Eastern Karaboro
- `xsb_Latn` — Sambal
- `xsm_Latn` — Kasem
- `xsu_Latn` — Sanumá
- `xta_Latn` — Alcozauca Mixtec
- `xtd_Latn` — Diuxi-Tilantongo Mixtec
- `xte_Latn` — Ketengban
- `xti_Latn` — Sinicahua Mixtec
- `xtm_Latn` — Magdalena Peñasco Mixtec
- `xtn_Latn` — Northern Tlaxiaco Mixtec
- `xtu_Latn` — Cuyamecalco Mixtec
- `xuo_Latn` — Kuo
- `yaa_Latn` — Yaminahua
- `yad_Latn` — Yagua
- `yal_Latn` — Yalunka
- `yam_Latn` — Yamba
- `yao_Latn` — Yao
- `yaq_Latn` — Yaqui
- `yas_Latn` — Nugunu (Cameroon)
- `yat_Latn` — Yambeta
- `yav_Latn` — Yangben
- `yay_Latn` — Agwagwune
- `yaz_Latn` — Lokaa
- `yba_Latn` — Yala
- `ybb_Latn` — Yemba
- `ycl_Latn` — Lolopo
- `ycn_Latn` — Yucuna
- `yer_Latn` — Tarok
- `yes_Latn` — Nyankpa
- `yka_Latn` — Yakan
- `yli_Latn` — Angguruk Yali
- `yor_Latn` — Yoruba — Nigeria
- `yre_Latn` — Yaouré
- `yua_Latn` — Yucateco
- `yuz_Latn` — Yuracare
- `yva_Latn` — Yawa
- `zaa_Latn` — Sierra de Juárez Zapotec
- `zab_Latn` — Western Tlacolula Valley Zapotec
- `zac_Latn` — Ocotlán Zapotec
- `zad_Latn` — Cajonos Zapotec
- `zae_Latn` — Yareni Zapotec
- `zai_Latn` — Isthmus Zapotec
- `zam_Latn` — Miahuatlán Zapotec
- `zao_Latn` — Ozolotepec Zapotec
- `zaq_Latn` — Aloápam Zapotec
- `zar_Latn` — Rincón Zapotec
- `zas_Latn` — Santo Domingo Albarradas Zapotec
- `zav_Latn` — Yatzachi Zapotec
- `zaw_Latn` — Mitla Zapotec
- `zca_Latn` — Coatecas Altas Zapotec
- `zga_Latn` — Kinga
- `zim_Latn` — Mesme
- `ziw_Latn` — Zigula
- `zmz_Latn` — Mbandja
- `zne_Latn` — Zande (individual language)
- `zoc_Latn` — Copainalá Zoque
- `zoh_Latn` — Chimalapa Zoque
- `zor_Latn` — Rayón Zoque
- `zos_Latn` — Francisco León Zoque
- `zpc_Latn` — Choapan Zapotec
- `zpg_Latn` — Guevea De Humboldt Zapotec
- `zpi_Latn` — Santa María Quiegolani Zapotec
- `zpl_Latn` — Lachixío Zapotec
- `zpm_Latn` — Mixtepec Zapotec
- `zpo_Latn` — Amatlán Zapotec
- `zpt_Latn` — San Vicente Coatlán Zapotec
- `zpu_Latn` — Yalálag Zapotec
- `zpv_Latn` — Chichicapan Zapotec
- `zpy_Latn` — Mazaltepec Zapotec
- `zpz_Latn` — Texmelucan Zapotec
- `zsm_Latn` — Standard Malay
- `ztg_Latn` — Xanaguía Zapotec
- `ztn_Latn` — Santa Catarina Albarradas Zapotec
- `ztp_Latn` — Loxicha Zapotec
- `ztq_Latn` — Quioquitani-Quierí Zapotec
- `zts_Latn` — Tilquiapan Zapotec
- `ztu_Latn` — Güilá Zapotec
- `zty_Latn` — Yatee Zapotec
- `zul_Latn` — Zulu — South Africa
- `zyb_Latn` — Yongbei Zhuang
- `zyp_Latn` — Zyphe Chin
- `zza_Latn` — Zaza

### Arabic (Arab) — 70 languages

- `acm_Arab` — Mesopotamian Arabic
- `acw_Arab` — Hijazi Arabic
- `aeb_Arab` — Tunisian Arabic
- `aec_Arab` — Saidi Arabic
- `afb_Arab` — Gulf Arabic
- `apc_Arab` — North Levantine Arabic
- `apd_Arab` — Sudanese Arabic
- `arb_Arab` — Standard Arabic
- `arq_Arab` — Algerian Arabic
- `ars_Arab` — Najdi Arabic
- `ary_Arab` — Moroccan Arabic
- `arz_Arab` — Egyptian Arabic
- `ayl_Arab` — Libyan Arabic
- `ayp_Arab` — North Mesopotamian Arabic
- `aze_Arab` — Azerbaijani — Azerbaijan
- `bcc_Arab` — Southern Balochi
- `bft_Arab` — Balti
- `bgp_Arab` — Eastern Balochi
- `bqi_Arab` — Bakhtiari
- `brh_Arab` — Brahui
- `bsh_Arab` — Kati
- `btv_Arab` — Bateri
- `ckb_Arab` — Central Kurdish
- `dcc_Arab` — Deccan
- `dmk_Arab` — Domaaki
- `dml_Arab` — Dameli
- `fas_Arab` — Persian — Iran, Islamic Republic of
- `ggg_Arab` — Gurgula
- `gig_Arab` — Goaria
- `gjk_Arab` — Kachi Koli
- `gju_Arab` — Gujari
- `glk_Arab` — Gilaki
- `gwc_Arab` — Gawri
- `gwt_Arab` — Gawar-Bati
- `hno_Arab` — Northern Hindko
- `kas_Arab` — Kashmiri — India
- `khw_Arab` — Khowar
- `kmr_Arab` — Northern Kurdish
- `kur_Arab` — Kurdish — Türkiye
- `kvx_Arab` — Parkari Koli
- `kxp_Arab` — Wadiyara Koli
- `lrk_Arab` — Loarki
- `lss_Arab` — Lasi
- `mki_Arab` — Dhatki
- `mve_Arab` — Marwari (Pakistan)
- `mvy_Arab` — Indus Kohistani
- `odk_Arab` — Od
- `oru_Arab` — Ormuri
- `pbt_Arab` — Southern Pashto
- `pbu_Arab` — Northern Pashto
- `phl_Arab` — Phalura
- `phr_Arab` — Pahari-Potwari
- `plk_Arab` — Kohistani Shina
- `pnb_Arab` — Western Panjabi
- `pst_Arab` — Central Pashto
- `pus_Arab` — Pashto — Afghanistan
- `rif_Arab` — Riffian
- `sbn_Arab` — Sindhi Bhil
- `scl_Arab` — Shina
- `skr_Arab` — Saraiki
- `snd_Arab` — Sindhi — Pakistan
- `ssi_Arab` — Sansi
- `trw_Arab` — Torwali
- `tuk_Arab` — Turkmen — Turkmenistan
- `uig_Arab` — Uyghur — China
- `urd_Arab` — Urdu — Pakistan
- `ush_Arab` — Ushojo
- `xhe_Arab` — Khetrani
- `xka_Arab` — Kalkoti
- `ydg_Arab` — Yidgha

### Devanagari (Nagari) (Deva) — 65 languages

- `anp_Deva` — Angika
- `awa_Deva` — Awadhi
- `bfy_Deva` — Bagheli
- `bfz_Deva` — Mahasu Pahari
- `bgc_Deva` — Haryanvi
- `bgq_Deva` — Bagri
- `bgw_Deva` — Bhatri
- `bha_Deva` — Bharia
- `bhb_Deva` — Bhili
- `bho_Deva` — Bhojpuri
- `bht_Deva` — Bhattiyali
- `bjj_Deva` — Kanauji
- `bns_Deva` — Bundeli
- `bra_Deva` — Braj
- `brx_Deva` — Bodo
- `cdj_Deva` — Churahi
- `dgo_Deva` — Dogri (individual language)
- `dhi_Deva` — Dhimal
- `dty_Deva` — Dotyali
- `fmu_Deva` — Far Western Muria
- `gbk_Deva` — Gaddi
- `gbm_Deva` — Garhwali
- `gom_Deva` — Goan Konkani
- `hin_Deva` — Hindi — India
- `hlb_Deva` — Halbi
- `hne_Deva` — Chhattisgarhi
- `kfb_Deva` — Northwestern Kolami
- `kfk_Deva` — Kinnauri
- `kfq_Deva` — Korku
- `kfx_Deva` — Kullu Pahari
- `kle_Deva` — Kulung (Nepal)
- `knn_Deva` — Konkani (individual language)
- `kru_Deva` — Kurukh
- `ksz_Deva` — Kodaku
- `lif_Deva` — Limbu
- `mag_Deva` — Magahi
- `mai_Deva` — Maithili
- `mar_Deva` — Marathi — India
- `mjl_Deva` — Mandeali
- `mrr_Deva` — Maria (India)
- `mtr_Deva` — Mewari
- `mup_Deva` — Malvi
- `nep_Deva` — Nepali — Nepal
- `new_Deva` — Newari
- `noe_Deva` — Nimadi
- `rav_Deva` — Sampang
- `rjs_Deva` — Rajbanshi
- `rwr_Deva` — Marwari (India)
- `sck_Deva` — Sadri
- `sgj_Deva` — Surgujia
- `sjp_Deva` — Surjapuri
- `srx_Deva` — Sirmauri
- `suz_Deva` — Sunwar
- `swv_Deva` — Shekhawati
- `taj_Deva` — Eastern Tamang
- `the_Deva` — Chitwania Tharu
- `thl_Deva` — Dangaura Tharu
- `thq_Deva` — Kochila Tharu
- `thr_Deva` — Rana Tharu
- `tkt_Deva` — Kathoriya Tharu
- `urd_Deva` — Urdu — Pakistan
- `vah_Deva` — Varhadi-Nagpuri
- `wbr_Deva` — Wagdi
- `xnr_Deva` — Kangri
- `xsr_Deva` — Sherpa

### Cyrillic (Cyrl) — 51 languages

- `abk_Cyrl` — Abkhazian — Georgia
- `ady_Cyrl` — Adyghe
- `agx_Cyrl` — Aghul
- `alt_Cyrl` — Southern Altai
- `ava_Cyrl` — Avaric — Russian Federation
- `aze_Cyrl` — Azerbaijani — Azerbaijan
- `bak_Cyrl` — Bashkir — Russian Federation
- `bel_Cyrl` — Belarusian — Belarus
- `bhh_Cyrl` — Bukharic
- `bul_Cyrl` — Bulgarian — Bulgaria
- `che_Cyrl` — Chechen — Russian Federation
- `chv_Cyrl` — Chuvash — Russian Federation
- `cjs_Cyrl` — Shor
- `ckt_Cyrl` — Chukot
- `crh_Cyrl` — Crimean Tatar
- `dar_Cyrl` — Dargwa
- `evn_Cyrl` — Evenki
- `gag_Cyrl` — Gagauz
- `gld_Cyrl` — Nanai
- `itl_Cyrl` — Itelmen
- `kaa_Cyrl` — Kara-Kalpak
- `kaz_Cyrl` — Kazakh — Kazakhstan
- `kbd_Cyrl` — Kabardian
- `kca_Cyrl` — Khanty
- `khk_Cyrl` — Halh Mongolian
- `kir_Cyrl` — Kyrgyz — Kyrgyzstan
- `kjh_Cyrl` — Khakas
- `kmr_Cyrl` — Northern Kurdish
- `kpv_Cyrl` — Komi-Zyrian
- `kpy_Cyrl` — Koryak
- `krc_Cyrl` — Karachay-Balkar
- `kum_Cyrl` — Kumyk
- `mhr_Cyrl` — Eastern Mari
- `mkd_Cyrl` — Macedonian — North Macedonia
- `mon_Cyrl` — Mongolian — Mongolia
- `mrj_Cyrl` — Western Mari
- `myv_Cyrl` — Erzya
- `nog_Cyrl` — Nogai
- `oss_Cyrl` — Ossetic — Georgia
- `rmc_Cyrl` — Carpathian Romani
- `rmy_Cyrl` — Vlax Romani
- `rus_Cyrl` — Russian — Russian Federation
- `sah_Cyrl` — Yakut
- `srp_Cyrl` — Serbian — Serbia
- `tat_Cyrl` — Tatar — Russian Federation
- `tgk_Cyrl` — Tajik — Tajikistan
- `udm_Cyrl` — Udmurt
- `uig_Cyrl` — Uyghur — China
- `ukr_Cyrl` — Ukrainian — Ukraine
- `uzb_Cyrl` — Uzbek — Uzbekistan
- `xal_Cyrl` — Kalmyk

### Ethiopic (Geʻez) (Ethi) — 10 languages

- `amh_Ethi` — Amharic — Ethiopia
- `guk_Ethi` — Gumuz
- `kqy_Ethi` — Koorete
- `ktb_Ethi` — Kambaata
- `kxc_Ethi` — Konso
- `mdy_Ethi` — Male (Ethiopia)
- `sgw_Ethi` — Sebat Bet Gurage
- `tig_Ethi` — Tigre
- `tir_Ethi` — Tigrinya — Ethiopia
- `wal_Ethi` — Wolaytta

### Thai (Thai) — 9 languages

- `bzi_Thai` — Bisu
- `kxm_Thai` — Northern Khmer
- `lcp_Thai` — Western Lawa
- `nod_Thai` — Northern Thai
- `pce_Thai` — Ruching Palaung
- `prt_Thai` — Phai
- `pww_Thai` — Pwo Northern Karen
- `tha_Thai` — Thai — Thailand
- `urk_Thai` — Urak Lawoi'

### Bengali (Beng) — 8 languages

- `asm_Beng` — Assamese — India
- `ben_Beng` — Bangla — Bangladesh
- `bng_Beng` — Benga
- `ctg_Beng` — Chittagonian
- `grt_Beng` — Garo
- `mni_Beng` — Manipuri
- `rah_Beng` — Rabha
- `rkt_Beng` — Rangpuri

### Tibetan (Tibt) — 6 languages

- `adx_Tibt` — Amdo Tibetan
- `bod_Tibt` — Tibetan — China
- `dzo_Tibt` — Dzongkha — Bhutan
- `khg_Tibt` — Khams Tibetan
- `lbj_Tibt` — Ladakhi
- `sip_Tibt` — Sikkimese

### Malayalam (Mlym) — 5 languages

- `mal_Mlym` — Malayalam — India
- `mjv_Mlym` — Mannan
- `muv_Mlym` — Muthuvan
- `tcy_Mlym` — Tulu
- `yea_Mlym` — Ravula

### Telugu (Telu) — 5 languages

- `gau_Telu` — Mudhili Gadaba
- `key_Telu` — Kupia
- `kff_Telu` — Koya
- `tel_Telu` — Telugu — India
- `wsg_Telu` — Adilabad Gondi

### Georgian (Mkhedruli) (Geor) — 4 languages

- `bbl_Geor` — Bats
- `kat_Geor` — Georgian — Georgia
- `sva_Geor` — Svan
- `xmf_Geor` — Mingrelian

### Han (Simplified variant) (Hans) — 4 languages

- `cdo_Hans` — Min Dong Chinese
- `cmn_Hans` — Mandarin Chinese
- `cpx_Hans` — Pu-Xian Chinese
- `yue_Hans` — Cantonese

### Khmer (Khmr) — 4 languages

- `cmo_Khmr` — Central Mnong
- `kdt_Khmr` — Kuy
- `khm_Khmr` — Khmer — Cambodia
- `krr_Khmr` — Krung

### Oriya (Orya) — 4 languages

- `hoc_Orya` — Ho
- `jun_Orya` — Juang
- `ory_Orya` — Odia (individual language)
- `uki_Orya` — Kui (India)

### Greek (Grek) — 3 languages

- `ell_Grek` — Greek — Greece
- `ell_Grek_cypr1249` — Greek — cypr1249 dialect — Greece
- `grc_Grek` — Ancient Greek

### Myanmar (Burmese) (Mymr) — 3 languages

- `mnw_Mymr` — Mon
- `mya_Mymr` — Burmese — Myanmar
- `shn_Mymr` — Shan

### Armenian (Armn) — 2 languages

- `hye_Armn` — Armenian — Armenia
- `hyw_Armn` — Western Armenian

### Unified Canadian Aboriginal Syllabics (Cans) — 2 languages

- `crk_Cans` — Plains Cree
- `ojb_Cans` — Northwestern Ojibwa

### Gujarati (Gujr) — 2 languages

- `guj_Gujr` — Gujarati — India
- `kfr_Gujr` — Kachhi

### Han (Traditional variant) (Hant) — 2 languages

- `cmn_Hant` — Mandarin Chinese
- `yue_Hant` — Cantonese

### Hebrew (Hebr) — 2 languages

- `heb_Hebr` — Hebrew — Israel
- `ydd_Hebr` — Eastern Yiddish

### Tamil (Taml) — 2 languages

- `tam_Taml` — Tamil — India
- `xua_Taml` — Alu Kurumba

### Tifinagh (Berber) (Tfng) — 2 languages

- `thv_Tfng` — Tahaggart Tamahaq
- `ttq_Tfng` — Tawallammat Tamajaq

### Gurmukhi (Guru) — 1 languages

- `pan_Guru` — Punjabi — India

### Hangul (Hangŭl, Hangeul) (Hang) — 1 languages

- `kor_Hang` — Korean — Korea, Republic of

### Japanese (alias for Han + Hiragana + Katakana) (Jpan) — 1 languages

- `jpn_Jpan` — Japanese — Japan

### Kayah Li (Kali) — 1 languages

- `kyu_Kali` — Western Kayah

### Kannada (Knda) — 1 languages

- `kan_Knda` — Kannada — India

### Lao (Laoo) — 1 languages

- `lao_Laoo` — Lao — Lao People's Democratic Republic

### Lisu (Fraser) (Lisu) — 1 languages

- `lis_Lisu` — Lisu

### Sinhala (Sinh) — 1 languages

- `sin_Sinh` — Sinhala — Sri Lanka

### Thaana (Thaa) — 1 languages

- `div_Thaa` — Divehi — Maldives

</details>
