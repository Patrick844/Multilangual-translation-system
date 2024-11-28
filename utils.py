model_dict = {
    """
    A dictionary mapping ISO 639-1 language codes to lists of model names for translation.

    Each key in the dictionary represents a source language, and the value is a list of
    Helsinki-NLP MarianMT model names. These models are used for translating between
    the source language and Arabic, or in the case of Romanian, to and from English
    and Arabic.

    Structure:
    - For direct translations: Lists contain two models for the same source-target pair.
    - For indirect translations (e.g., Romanian): Lists include intermediate English models.

    Example:
        model_dict = {
            "eng": ["Helsinki-NLP/opus-mt-en-ar", "Helsinki-NLP/opus-mt-en-ar"],
            "ron": [
                "Helsinki-NLP/opus-mt-roa-en",
                "Helsinki-NLP/opus-mt-roa-en",
                "Helsinki-NLP/opus-mt-en-ar",
                "Helsinki-NLP/opus-mt-en-ar",
            ],
        }

    Keys:
        - ISO 639-1 codes for supported languages (e.g., "eng" for English, "fra" for French).
    Values:
        - Lists of MarianMT model names for translation.

    """
    "eng": ["Helsinki-NLP/opus-mt-en-ar","Helsinki-NLP/opus-mt-en-ar"],
    "fra": ["Helsinki-NLP/opus-mt-fr-ar","Helsinki-NLP/opus-mt-fr-ar"],
    "ita": ["Helsinki-NLP/opus-mt-it-ar","Helsinki-NLP/opus-mt-it-ar"],
    "ron":["Helsinki-NLP/opus-mt-roa-en","Helsinki-NLP/opus-mt-roa-en","Helsinki-NLP/opus-mt-en-ar","Helsinki-NLP/opus-mt-en-ar"],
    "rus":["Helsinki-NLP/opus-mt-ru-ar","Helsinki-NLP/opus-mt-ru-ar"],
    "tur":["Helsinki-NLP/opus-mt-tr-ar","Helsinki-NLP/opus-mt-tr-ar"],
    "spa":["Helsinki-NLP/opus-mt-es-ar","Helsinki-NLP/opus-mt-es-ar"],
    "ell":["Helsinki-NLP/opus-mt-el-ar","Helsinki-NLP/opus-mt-el-ar"]
}




lang_code = {
    """
    A dictionary mapping ISO 639-1 language codes to ISO 639-3 language codes.

    This dictionary provides a conversion between two widely used language code formats.
    ISO 639-1 codes are two-letter codes, while ISO 639-3 codes are three-letter codes
    used for more precise identification of languages.

    Structure:
        - Keys: ISO 639-1 two-letter language codes (e.g., "en", "fr", "es").
        - Values: ISO 639-3 three-letter language codes (e.g., "eng", "fra", "spa").

    Example:
        lang_code = {
            "en": "eng",  # English
            "fr": "fra",  # French
            "es": "spa",  # Spanish
            "ar": "ara",  # Arabic
        }

    Usage:
        - Convert language codes for compatibility with various APIs or datasets.
    """
    'aa': 'aar', 'ab': 'abk', 'af': 'afr', 'ak': 'aka', 'am': 'amh',
    'ar': 'ara', 'an': 'arg', 'as': 'asm', 'av': 'ava', 'ay': 'aym',
    'az': 'aze', 'ba': 'bak', 'be': 'bel', 'bg': 'bul', 'bh': 'bih',
    'bi': 'bis', 'bm': 'bam', 'bn': 'ben', 'bo': 'bod', 'br': 'bre',
    'bs': 'bos', 'ca': 'cat', 'ce': 'che', 'ch': 'cha', 'co': 'cos',
    'cr': 'cre', 'cs': 'ces', 'cu': 'chu', 'cv': 'chv', 'cy': 'cym',
    'da': 'dan', 'de': 'deu', 'dv': 'div', 'dz': 'dzo', 'ee': 'ewe',
    'el': 'ell', 'en': 'eng', 'eo': 'epo', 'es': 'spa', 'et': 'est',
    'eu': 'eus', 'fa': 'fas', 'ff': 'ful', 'fi': 'fin', 'fj': 'fij',
    'fo': 'fao', 'fr': 'fra', 'fy': 'fry', 'ga': 'gle', 'gd': 'gla',
    'gl': 'glg', 'gn': 'grn', 'gu': 'guj', 'gv': 'glv', 'ha': 'hau',
    'he': 'heb', 'hi': 'hin', 'ho': 'hmo', 'hr': 'hrv', 'ht': 'hat',
    'hu': 'hun', 'hy': 'hye', 'hz': 'her', 'ia': 'ina', 'id': 'ind',
    'ie': 'ile', 'ig': 'ibo', 'ii': 'iii', 'ik': 'ipk', 'io': 'ido',
    'is': 'isl', 'it': 'ita', 'iu': 'iku', 'ja': 'jpn', 'jv': 'jav',
    'ka': 'kat', 'kg': 'kon', 'ki': 'kik', 'kj': 'kua', 'kk': 'kaz',
    'kl': 'kal', 'km': 'khm', 'kn': 'kan', 'ko': 'kor', 'kr': 'kau',
    'ks': 'kas', 'ku': 'kur', 'kv': 'kom', 'kw': 'cor', 'ky': 'kir',
    'la': 'lat', 'lb': 'ltz', 'lg': 'lug', 'li': 'lim', 'ln': 'lin',
    'lo': 'lao', 'lt': 'lit', 'lu': 'lub', 'lv': 'lav', 'mg': 'mlg',
    'mh': 'mah', 'mi': 'mri', 'mk': 'mkd', 'ml': 'mal', 'mn': 'mon',
    'mr': 'mar', 'ms': 'msa', 'mt': 'mlt', 'my': 'mya', 'na': 'nau',
    'nb': 'nob', 'nd': 'nde', 'ne': 'nep', 'ng': 'ndo', 'nl': 'nld',
    'nn': 'nno', 'no': 'nor', 'nr': 'nbl', 'nv': 'nav', 'ny': 'nya',
    'oc': 'oci', 'oj': 'oji', 'om': 'orm', 'or': 'ori', 'os': 'oss',
    'pa': 'pan', 'pi': 'pli', 'pl': 'pol', 'ps': 'pus', 'pt': 'por',
    'qu': 'que', 'rm': 'roh', 'rn': 'run', 'ro': 'ron', 'ru': 'rus',
    'rw': 'kin', 'sa': 'san', 'sc': 'srd', 'sd': 'snd', 'se': 'sme',
    'sg': 'sag', 'si': 'sin', 'sk': 'slk', 'sl': 'slv', 'sm': 'smo',
    'sn': 'sna', 'so': 'som', 'sq': 'sqi', 'sr': 'srp', 'ss': 'ssw',
    'st': 'sot', 'su': 'sun', 'sv': 'swe', 'sw': 'swa', 'ta': 'tam',
    'te': 'tel', 'tg': 'tgk', 'th': 'tha', 'ti': 'tir', 'tk': 'tuk',
    'tl': 'tgl', 'tn': 'tsn', 'to': 'ton', 'tr': 'tur', 'ts': 'tso',
    'tt': 'tat', 'tw': 'twi', 'ty': 'tah', 'ug': 'uig', 'uk': 'ukr',
    'ur': 'urd', 'uz': 'uzb', 've': 'ven', 'vi': 'vie', 'vo': 'vol',
    'wa': 'wln', 'wo': 'wol', 'xh': 'xho', 'yi': 'yid', 'yo': 'yor',
    'za': 'zha', 'zh': 'zho', 'zu': 'zul'
}


# Updated comprehensive dictionary of medical and clinical abbreviations in lowercase

abbreviation_dict = {
    """
    A dictionary mapping medical and clinical abbreviations to their expanded meanings.

    This dictionary is designed to support text normalization and preprocessing for
    medical and clinical data. Each key represents a commonly used abbreviation, and
    the corresponding value is the expanded, human-readable form.

    Structure:
        - Keys: Medical abbreviations in lowercase (e.g., "bp", "hr", "copd").
        - Values: Expanded meanings of the abbreviations.

    Example:
        abbreviation_dict = {
            "bp": "blood pressure",
            "hr": "heart rate",
            "copd": "chronic obstructive pulmonary disease",
            "dm": "diabetes mellitus",
        }

    Usage:
        - Expand abbreviations in medical datasets for improved text readability and analysis.
        - Facilitate preprocessing in machine learning pipelines for clinical text data.

    Notes:
        - Abbreviations and their meanings are provided in lowercase for consistency.
        - Commonly used acronyms across various medical domains are included.
    """
    'bp': 'blood pressure',
    'hr': 'heart rate',
    'rr': 'respiratory rate',
    'o2 sat': 'oxygen saturation',
    'copd': 'chronic obstructive pulmonary disease',
    'dm': 'diabetes mellitus',
    'chf': 'congestive heart failure',
    'mi': 'myocardial infarction (heart attack)',
    'htn': 'hypertension',
    'tia': 'transient ischemic attack (mini-stroke)',
    'cad': 'coronary artery disease',
    'ckd': 'chronic kidney disease',
    'esrd': 'end-stage renal disease',
    'pvd': 'peripheral vascular disease',
    'sob': 'shortness of breath',
    'gi': 'gastrointestinal',
    'gu': 'genitourinary',
    'n/v': 'nausea and vomiting',
    'ua': 'urinalysis',
    'cxr': 'chest x-ray',
    'ecg': 'electrocardiogram',
    'eeg': 'electroencephalogram',
    'cbc': 'complete blood count',
    'bun': 'blood urea nitrogen',
    'abg': 'arterial blood gas',
    'wbc': 'white blood cell',
    'rbc': 'red blood cell',
    'hgb': 'hemoglobin',
    'hct': 'hematocrit',
    'inr': 'international normalized ratio',
    'pt': 'prothrombin time',
    'ptt': 'partial thromboplastin time',
    'ct': 'computed tomography',
    'mri': 'magnetic resonance imaging',
    'iv': 'intravenous',
    'po': 'by mouth (per os)',
    'npo': 'nothing by mouth (nil per os)',
    'bid': 'twice a day (bis in die)',
    'tid': 'three times a day (ter in die)',
    'qd': 'once a day (quaque die)',
    'prn': 'as needed (pro re nata)',
    'dx': 'diagnosis',
    'rx': 'prescription',
    'tx': 'treatment',
    'px': 'prognosis',
    'c/o': 'complains of',
    'h/o': 'history of',
    's/p': 'status post (after)',
    'o/e': 'on examination',
    'r/o': 'rule out',
    'i&d': 'incision and drainage',
    'bph': 'benign prostatic hyperplasia',
    'uri': 'upper respiratory infection',
    'uti': 'urinary tract infection',
    'dm2': 'type 2 diabetes mellitus',
    'hba1c': 'hemoglobin a1c',
    'lft': 'liver function test',
    'dvt': 'deep vein thrombosis',
    'pe': 'pulmonary embolism',
    'nsr': 'normal sinus rhythm',
    'afib': 'atrial fibrillation',
    'ards': 'acute respiratory distress syndrome',
    'icu': 'intensive care unit',
    'er': 'emergency room',
    'opd': 'outpatient department',
    'or': 'operating room',
    'cns': 'central nervous system',
    'pns': 'peripheral nervous system',
    'tbi': 'traumatic brain injury',
    'ms': 'multiple sclerosis',
    'als': 'amyotrophic lateral sclerosis',
    'ck': 'creatine kinase',
    'crp': 'c-reactive protein',
    'esr': 'erythrocyte sedimentation rate',
    'ldh': 'lactate dehydrogenase',
    'acei': 'angiotensin converting enzyme inhibitor',
    'arb': 'angiotensin ii receptor blocker',
    'ccb': 'calcium channel blocker',
    'nsaid': 'non-steroidal anti-inflammatory drug',
    'asa': 'aspirin (acetylsalicylic acid)',
    'plavix': 'clopidogrel',
    'statin': 'cholesterol-lowering medication',
    'hrt': 'hormone replacement therapy',
    'ssri': 'selective serotonin reuptake inhibitor',
    'nrt': 'nicotine replacement therapy',
    'ppi': 'proton pump inhibitor',
    'gerd': 'gastroesophageal reflux disease',
    'ibs': 'irritable bowel syndrome',
    'ibd': 'inflammatory bowel disease',
    'uc': 'ulcerative colitis',
    'crohn’s': 'crohn’s disease',
    'gfr': 'glomerular filtration rate',
    'std': 'sexually transmitted disease',
    'hiv': 'human immunodeficiency virus',
    'aids': 'acquired immunodeficiency syndrome',
    'tb': 'tuberculosis',
    'csf': 'cerebrospinal fluid',
    'tia': 'transient ischemic attack',
    'acl': 'anterior cruciate ligament',
    'pcl': 'posterior cruciate ligament',
    'mcl': 'medial collateral ligament',
    'lcl': 'lateral collateral ligament',
    'bmi': 'body mass index',
    'etoh': 'alcohol (ethanol)',
    'tbi': 'traumatic brain injury',
    'loc': 'loss of consciousness',
    'sx': 'symptoms',
    'dx': 'diagnosis',
    'rx': 'prescription',
    'otc': 'over the counter',
    'adr': 'adverse drug reaction',
    'cpr': 'cardiopulmonary resuscitation',
    'cabg': 'coronary artery bypass graft',
    'pci': 'percutaneous coronary intervention',
    'stemi': 'st-elevation myocardial infarction',
    'nstemi': 'non-st elevation myocardial infarction',
    'icd': 'implantable cardioverter defibrillator',
    'pad': 'peripheral artery disease',
    'vte': 'venous thromboembolism',
    'aki': 'acute kidney injury',
    'cap': 'community acquired pneumonia',
    'hap': 'hospital acquired pneumonia',
    'vap': 'ventilator associated pneumonia',
    'rsv': 'respiratory syncytial virus',
    'fna': 'fine needle aspiration',
    'bmt': 'bone marrow transplant',
    'prn': 'as needed',
    'hs': 'at bedtime',
    'q.d.': 'every day',
    't.i.d.': 'three times a day',
    'b.i.d.': 'twice a day',
    'q.i.d.': 'four times a day',
    'q.h.': 'every hour',
    'ad lib.': 'as desired',
    'qhs': 'every night at bedtime',
    'qid': 'four times a day',
    'npo': 'nothing by mouth',
    'ivf': 'intravenous fluids',
    'dnr': 'do not resuscitate',
    'perrla': 'pupils equal, round, reactive to light and accommodation',
    'pft': 'pulmonary function test',
    'll': 'lower limb',
    'abd': 'abdomen'
}