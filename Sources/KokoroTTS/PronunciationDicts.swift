import Foundation

/// Pronunciation dictionaries for Kokoro TTS multilingual support.
///
/// Large dictionaries (French, Portuguese, Hindi) are loaded from JSON resource
/// files at runtime. Smaller ones (Spanish, Italian, German, Korean) are
/// embedded as Swift literals. Source: ipa-dict (MIT, open-dict-data/ipa-dict)
/// and standard phonetic references.
enum PronunciationDicts {

    // MARK: - JSON Resource Loading

    private static func loadJSON(_ name: String) -> [String: String] {
        let url = Bundle.module.url(forResource: name, withExtension: "json")
                  ?? Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Resources")
        guard let url else { return [:] }
        guard let data = try? Data(contentsOf: url),
              let dict = try? JSONDecoder().decode([String: String].self, from: data) else {
            return [:]
        }
        return dict
    }

    // MARK: - French (3093 entries, JSON)

    static let fr: [String: String] = loadJSON("dict_fr")

    // MARK: - Portuguese (2001 entries, JSON)

    static let pt: [String: String] = loadJSON("dict_pt")

    // MARK: - Hindi (146 entries, JSON)

    static let hi: [String: String] = loadJSON("dict_hi")

    // MARK: - Spanish (125 entries)

    static let es: [String: String] = [
        "adiós": "aðjˈos", "agua": "ˈaɣwa", "ahora": "aˈoɾa", "al": "ˈal",
        "alto": "ˈalto", "amarillo": "ˌamaɾˈiʎo", "amigo": "amˈiɣo",
        "antes": "ˈantes", "aquí": "akˈi", "azul": "aθˈul", "año": "ˈaɲo",
        "bajo": "bˈaxo", "bien": "bjˈen", "blanco": "blˈanko", "boca": "bˈoka",
        "brazo": "bɾˈaθo", "bueno": "bwˈeno", "buenos": "bwˈenos",
        "cabeza": "kaβˈeθa", "café": "kafˈe", "calle": "kˈaʎe", "casa": "kˈasa",
        "cinco": "θˈinko", "ciudad": "θjuðˈad", "comida": "komˈiða",
        "como": "kˈomo", "con": "kˈon", "corazón": "kˌoɾaθˈon", "cosa": "kˈosa",
        "cuando": "kwˈando", "cuatro": "kwˈatɾo", "dar": "dˈaɾ", "de": "dˈe",
        "decir": "deθˈiɾ", "del": "dˈel", "desde": "dˈesðe", "después": "despwˈes",
        "donde": "dˈonde", "dos": "dˈos", "día": "dˈia", "días": "dˈias",
        "el": "ˈel", "ella": "ˈeʎa", "ellas": "ˈeʎas", "ellos": "ˈeʎos",
        "en": "ˈen", "entre": "ˈɛntɾe", "eso": "ˈeso", "estar": "estˈaɾ",
        "esto": "ˈesto", "familia": "famˈilja", "gracias": "ɡɾˈaθjas",
        "grande": "ɡɾˈande", "haber": "aβˈeɾ", "hacer": "aθˈeɾ", "hasta": "ˈasta",
        "hermana": "eɾmˈana", "hermano": "eɾmˈano", "hija": "ˈixa",
        "hijo": "ˈixo", "hola": "ˈola", "hombre": "ˈombɾe", "ir": "ˈiɾ",
        "joven": "xˈoβen", "la": "lˈa", "las": "lˈas", "leche": "lˈetʃe",
        "llegar": "ʎeɣˈaɾ", "los": "lˈos", "madre": "mˈaðɾe", "malo": "mˈalo",
        "mano": "mˈano", "mesa": "mˈesa", "mujer": "muxˈeɾ", "mundo": "mˈundo",
        "muy": "mˈuj", "más": "mˈas", "negro": "nˈeɣɾo", "niña": "nˈiɲa",
        "niño": "nˈiɲo", "no": "nˈo", "nosotros": "nosˈotɾos", "nuevo": "nwˈeβo",
        "nunca": "nˈunka", "ojo": "ˈoxo", "padre": "pˈaðɾe", "pan": "pˈan",
        "para": "pˈaɾa", "país": "paˈis", "pequeño": "pekˈeɲo",
        "perdón": "peɾðˈon", "pero": "pˈeɾo", "pie": "pjˈe", "poder": "poðˈeɾ",
        "por": "pˈoɾ", "porque": "pˈoɾke", "prueba": "pɾuˈeβa",
        "puerta": "pwˈeɾta", "querer": "keɾˈeɾ", "qué": "kˈe", "rojo": "rˈoxo",
        "saber": "saβˈeɾ", "ser": "sˈer", "siempre": "sjˈempɾe", "sin": "sˈin",
        "sobre": "sˈoβɾe", "sí": "sˈi", "también": "tambjˈen", "tener": "tenˈeɾ",
        "tiempo": "tjˈempo", "todo": "tˈoðo", "tres": "tɾˈes", "tú": "tˈu",
        "un": "ˈun", "una": "ˈuna", "uno": "ˈuno", "usted": "ustˈed",
        "ventana": "bentˈana", "ver": "bˈeɾ", "verde": "bˈeɾðe", "vida": "bˈiða",
        "viejo": "bjˈexo", "vino": "bˈino", "yo": "ʝˈo", "él": "ˈel",
    ]

    // MARK: - Italian (174 entries)

    static let it: [String: String] = [
        "acqua": "ˈakːwa", "alto": "ˈalto", "altro": "ˈaltro", "amico": "amˈiko",
        "anche": "ˈanke", "andare": "andˈare", "anno": "ˈanno", "avere": "avˈere",
        "bambina": "bambˈina", "bambino": "bambˈino", "basso": "bˈasso",
        "bella": "bˈɛlla", "bello": "bˈɛllo", "bene": "bˈɛne", "bere": "bˈere",
        "bianco": "bjˈanko", "blu": "blˈu", "bocca": "bˈokːa",
        "braccio": "brˈatʃːo", "brutto": "brˈutːo", "buonasera": "bwˌɔnasˈera",
        "buongiorno": "bʊondʒˈɔrno", "buono": "bʊˈɔno", "caffè": "kaffˈɛ",
        "caldo": "kˈaldo", "casa": "kˈaza", "cattivo": "katːˈivo", "che": "kˈe",
        "chi": "kˈi", "ciao": "tʃˈao", "cibo": "tʃˈibo", "cinque": "tʃˈinkwe",
        "città": "tʃitːˈa", "come": "kˈome", "cosa": "kˈɔza", "cuore": "kʊˈɔre",
        "dal": "dˈal", "dalla": "dˈalla", "dare": "dˈare", "debole": "dˈebole",
        "dei": "dˈeɪ", "del": "dˈel", "della": "dˈella", "delle": "dˈelle",
        "dello": "dˈello", "di": "dˈi", "dieci": "djˈɛtʃɪ", "dire": "dˈire",
        "domani": "domˈanɪ", "donna": "dˈɔnna", "dopo": "dˈopo",
        "dormire": "dormˈire", "dove": "dˈove", "dovere": "dovˈere", "due": "dˈue",
        "erano": "ˈɛrano", "essere": "ˈɛssere", "famiglia": "famˈiʎa",
        "fare": "fˈare", "felice": "felˈitʃe", "figlia": "fˈiʎa",
        "figlio": "fˈiʎo", "finestra": "finˈɛstra", "forte": "fˈɔrte",
        "fratello": "fratˈɛllo", "freddo": "frˈedːo", "gamba": "ɡˈamba",
        "giallo": "dʒˈallo", "giorno": "dʒˈorno", "giovane": "dʒˈovane",
        "gli": "ʎˈɪ", "grande": "ɡrˈande", "grazie": "ɡrˈatsje",
        "ieri": "jˈɛrɪ", "il": "ˈiːl", "io": "ˈio", "la": "lˈa",
        "latte": "lˈatːe", "le": "lˈe", "leggere": "lˈɛdʒːere", "lei": "lˈɛi",
        "lo": "lˈo", "loro": "lˈɔro", "lui": "lˈui", "lungo": "lˈuŋɡo",
        "ma": "mˈa", "madre": "mˈadre", "mai": "mˈaj", "mangiare": "mandʒˈare",
        "mano": "mˈano", "mattina": "matːˈina", "migliore": "miʎˈore",
        "molto": "mˈolto", "mondo": "mˈondo", "nero": "nˈero", "noi": "nˈoi",
        "non": "nˈon", "notte": "nˈɔtːe", "nove": "nˈɔve", "nuovo": "nʊˈɔvo",
        "occhio": "ˈɔkːio", "oggi": "ˈɔdʒːɪ", "ogni": "ˈoɲɲɪ", "ora": "ˈora",
        "otto": "ˈɔtːo", "padre": "pˈadre", "paese": "paˈeze", "pane": "pˈane",
        "parlare": "parlˈare", "peggiore": "pedʒːˈore", "pensare": "pensˈare",
        "perché": "perkˈe", "piccolo": "pˈikːolo", "piede": "pjˈɛde",
        "più": "pjˈu", "porta": "pˈɔrta", "potere": "potˈere", "prego": "prˈɛɡo",
        "prima": "prˈima", "primo": "prˈimo", "prova": "prˈɔva",
        "quale": "kwˈale", "quando": "kwˈando", "quanto": "kwˈanto",
        "quattro": "kwˈatːro", "quello": "kwˈello", "questa": "kwˈesta",
        "questo": "kwˈesto", "qui": "kwˈi", "rosso": "rˈosso",
        "sapere": "sapˈere", "scrivere": "skrˈivere", "scusi": "skˈuzɪ",
        "secondo": "sekˈondo", "sedia": "sˈɛdia", "sei": "sˈɛi",
        "sempre": "sˈɛmpre", "sentire": "sentˈire", "sera": "sˈera",
        "sette": "sˈɛtːe", "siamo": "sjˈamo", "siete": "sjˈete",
        "sole": "sˈole", "sono": "sˈono", "sorella": "sorˈɛlla",
        "splende": "splˈɛnde", "stare": "stˈare", "stata": "stˈata",
        "stato": "stˈato", "stesso": "stˈesso", "strada": "strˈada",
        "sì": "sˈiː", "tavola": "tˈavola", "tempo": "tˈɛmpo",
        "terzo": "tˈɛrtso", "testa": "tˈɛsta", "tre": "trˈe",
        "triste": "trˈiste", "tu": "tˈu", "tutto": "tˈutːo",
        "ultimo": "ˈultimo", "un": "ˈun", "una": "ˈuna", "uno": "ˈuno",
        "uomo": "wˈɔmo", "vecchio": "vˈɛkːio", "vedere": "vedˈere",
        "venire": "venˈire", "verde": "vˈerde", "vino": "vˈino",
        "vita": "vˈita", "voi": "vˈoi", "volere": "volˈere", "è": "ˈɛː",
    ]

    // MARK: - German (142 entries)

    static let de: [String: String] = [
        "aber": "ˈɑːbɜ", "acht": "ˈaxt", "als": "ˈals", "alt": "ˈalt",
        "arbeit": "ˈaɾbaɪt", "arbeiten": "ˈaɾbaɪtən", "arm": "ˈaɾm", "auch": "ˈaʊx",
        "auge": "ˈaʊɡə", "bein": "bˈaɪn", "berg": "bˈɛɾk", "bitte": "bˈɪtə",
        "blau": "blˈaʊ", "brot": "bɾˈoːt", "bruder": "bɾˈuːdɜ", "danke": "dˈaŋkə",
        "das": "dˈas", "dass": "dˈas", "denken": "dˈɛŋkən", "denn": "dˈɛn",
        "der": "dˈɛɾ", "die": "dˈiː", "dort": "dˈɔɾt", "drei": "dɾˈaɪ", "du": "dˈuː",
        "ein": "ˈaɪn", "eine": "ˈaɪnə", "eins": "ˈaɪns", "er": "ˈɛɾ", "es": "ˈɛs",
        "essen": "ˈɛsən", "familie": "famˈiːlɪə", "fenster": "fˈɛnstɜ",
        "finden": "fˈɪndən", "frau": "frˈaʊ", "freund": "frˈɔønt", "fünf": "fˈʏnf",
        "geben": "ɡˈeːbən", "gehen": "ɡˈeːən", "gelb": "ɡˈɛlp", "geld": "ɡˈɛlt",
        "gestern": "ɡˈɛstɜn", "groß": "ɡɾˈoːs", "grün": "ɡɾˈyːn", "gut": "ɡˈuːt",
        "haben": "hˈɑːbən", "hallo": "hˈaloː", "hand": "hˈant", "haus": "hˈaʊs",
        "herz": "hˈɛɾts", "heute": "hˈɔøtə", "hier": "hˈiːɾ", "himmel": "hˈɪməl",
        "hoch": "hˈoːx", "ich": "ˈɪç", "ihr": "ˈiːɾ", "immer": "ˈɪmɜ", "ja": "jˈɑː",
        "jahr": "jˈɑːɾ", "jetzt": "jˈɛtst", "jung": "jˈʊŋ", "kaffee": "kˈafeː",
        "kaufen": "kˈaʊfən", "kind": "kˈɪnt", "klein": "klˈaɪn", "kommen": "kˈɔmən",
        "kopf": "kˈɔpf", "kurz": "kˈʊɐts", "können": "kˈœnən", "land": "lˈant",
        "lang": "lˈaŋ", "laufen": "lˈaʊfən", "leben": "lˈeːbən", "lesen": "lˈeːzən",
        "machen": "mˈaxən", "manchmal": "mˈançmɑːl", "mann": "mˈan", "meer": "mˈeːɾ",
        "milch": "mˈɪlç", "mond": "mˈoːnt", "morgen": "mˈɔɾɡən", "mund": "mˈʊnt",
        "mutter": "mˈʊtɜ", "müssen": "mˈʏsən", "name": "nˈɑːmə", "nehmen": "nˈeːmən",
        "nein": "nˈaɪn", "neu": "nˈɔø", "neun": "nˈɔøn", "nicht": "nˈɪçt", "nie": "nˈiː",
        "noch": "nˈɔx", "ob": "ˈɔp", "oder": "ˈoːdɜ", "oft": "ˈɔft", "rot": "rˈoːt",
        "schlafen": "ʃlˈɑːfən", "schlecht": "ʃlˈɛçt", "schon": "ʃˈoːn",
        "schreiben": "ʃrˈaɪbən", "schule": "ʃˈuːlə", "schwarz": "ʃvˈaɾts",
        "schwester": "ʃvˈɛstɜ", "sechs": "zˈɛks", "sehen": "zˈeːən", "sein": "zˈaɪn",
        "sie": "zˈiː", "sieben": "zˈiːbən", "sohn": "zˈoːn", "sollen": "zˈɔlən",
        "sonne": "zˈɔnə", "spielen": "ʃpˈiːlən", "sprechen": "ʃpɾˈɛçən",
        "stadt": "ʃtˈat", "tag": "tˈɑːk", "tisch": "tˈɪʃ", "tochter": "tˈɔxtɜ",
        "trinken": "tɾˈɪŋkən", "tür": "tˈyːɾ", "und": "ˈʊnt", "vater": "fˈɑːtɜ",
        "vier": "fˈiːɾ", "wald": "vˈalt", "wann": "vˈan", "warum": "vɑːrˈʊm",
        "was": "vˈas", "wasser": "vˈasɜ", "weil": "vˈaɪl", "wein": "vˈaɪn",
        "weiß": "vˈaɪs", "welt": "vˈɛlt", "wenn": "vˈɛn", "wer": "vˈeːɾ",
        "werden": "vˈɛɾdən", "wie": "vˈiː", "wir": "vˈiːɾ", "wissen": "vˈɪsən",
        "wo": "vˈoː", "wollen": "vˈɔlən", "zehn": "tsˈeːn", "zeit": "tsˈaɪt",
        "zwei": "tsvˈaɪ",
    ]

    // MARK: - Korean (92 entries)

    static let ko: [String: String] = [
        "가": "ka", "가다": "kada", "가족": "kadʒok", "감사": "kamsa",
        "강": "kaŋ", "거기": "kʌki", "것": "kʌt", "그": "kɯ", "길": "kil",
        "나": "na", "나라": "naɾa", "날": "nal", "남자": "namdʒa", "내일": "nɛil",
        "너": "nʌ", "네": "ne", "넷": "net", "눈": "nun",
        "다섯": "dasʌt", "달": "dal", "돈": "don", "동생": "doŋsɛŋ",
        "되다": "dweda", "둘": "dul", "딸": "tal",
        "마시다": "masida", "마음": "maɯm", "머리": "mʌɾi", "먹다": "mʌkta",
        "문": "mun", "물": "mul", "미안": "mian",
        "바다": "pada", "발": "pal", "밥": "pap", "보다": "poda",
        "사람": "saɾam", "산": "san", "세계": "sekje", "셋": "set",
        "손": "son", "시간": "sikan", "쓰다": "sɯda",
        "아니": "ani", "아들": "adɯl", "아버지": "abʌdʒi", "아이": "ai",
        "아홉": "ahop", "안녕": "annjʌŋ", "알다": "alda",
        "어디": "ʌdi", "어머니": "ʌmʌni", "어제": "ʌdʒe", "언제": "ʌndʒe",
        "없다": "ʌpta", "여기": "jʌki", "여자": "jʌdʒa", "열": "jʌl",
        "오늘": "onɯl", "오다": "oda", "왜": "wɛ", "우리": "uɾi",
        "이": "i", "이름": "iɾɯm", "읽다": "ikta", "입": "ip", "있다": "itta",
        "자다": "tʃada", "작다": "tʃakta", "저": "tʃʌ", "적다": "tʃʌkta",
        "좋다": "tʃotta", "주다": "tʃuda", "지금": "tʃikɯm", "집": "tʃip",
        "차": "tʃa", "친구": "tʃinku", "크다": "kʰɯda",
        "하나": "hana", "하늘": "hanɯl", "하다": "hada", "학교": "hakkjo",
        "해": "hɛ", "형": "hjʌŋ", "회사": "hwesa",
    ]

}
