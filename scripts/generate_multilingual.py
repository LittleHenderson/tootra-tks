#!/usr/bin/env python3
"""
TKS Multi-Language Generator

NO HEBREW. NO KABBALAH. PURE TKS CANON.

Generates TKS concepts expressed in 15 different languages.
Also generates translation pairs between languages and English.

15 Languages:
- Romance: Spanish, French, Portuguese, Italian
- Germanic: German, Dutch, Swedish
- Slavic: Russian, Polish
- Asian: Japanese, Chinese, Korean, Hindi
- African: Swahili
- Semitic: Arabic

Usage:
    python scripts/generate_multilingual.py
"""

import json
import random
from pathlib import Path
from typing import List, Dict
import argparse

# =============================================================================
# TKS CANON - NO HEBREW
# =============================================================================

WORLDS = {
    "A": "Spiritual",
    "B": "Mental",
    "C": "Emotional",
    "D": "Physical",
}

POSITIONS = {
    "1": "Mind",
    "2": "Positive",
    "3": "Negative",
    "4": "Vibration",
    "5": "Female",
    "6": "Male",
    "7": "Rhythm",
    "8": "Cause",
    "9": "Effect",
    "10": "All",
}

OPERATORS = ["+T", "-T", "*T", "/T", "+", "-", "->", "<-", "o"]

# =============================================================================
# LANGUAGE VOCABULARIES - 15 LANGUAGES
# =============================================================================

LANGUAGES = {
    # =========================================================================
    # ROMANCE LANGUAGES
    # =========================================================================
    "spanish": {
        "name": "Spanish",
        "code": "es",
        "worlds": {
            "A": ["reino espiritual", "plano espiritual", "dimension espiritual"],
            "B": ["reino mental", "plano mental", "dimension mental"],
            "C": ["reino emocional", "plano emocional", "dimension emocional"],
            "D": ["reino fisico", "plano material", "dimension fisica"],
        },
        "positions": {
            "1": ["conciencia", "mente", "consciencia"],
            "2": ["polaridad positiva", "fuerza atractiva", "expansion"],
            "3": ["polaridad negativa", "fuerza repulsiva", "contraccion"],
            "4": ["vibracion", "frecuencia", "oscilacion"],
            "5": ["principio femenino", "receptividad", "entrada"],
            "6": ["principio masculino", "proyeccion", "salida"],
            "7": ["ritmo", "ciclo", "periodicidad"],
            "8": ["causa", "origen", "fuente"],
            "9": ["efecto", "resultado", "manifestacion"],
            "10": ["totalidad", "integridad", "todo"],
        },
        "operators": {
            "+T": ["acumula temporalmente con", "crece con el tiempo con"],
            "-T": ["disminuye temporalmente de", "reduce con el tiempo de"],
            "*T": ["amplifica temporalmente con", "resuena con"],
            "/T": ["atenua temporalmente a traves de", "dispersa con el tiempo"],
            "+": ["combina con", "se une con"],
            "-": ["separa de", "se diferencia de"],
            "->": ["fluye hacia", "se transforma en"],
            "<-": ["deriva de", "recibe de"],
            "o": ["cicla con", "circula a traves de"],
        },
        "templates": [
            "El {elem1} {op} el {elem2}",
            "{elem1} {op} {elem2} en este trabajo",
            "Esta ecuacion muestra {elem1} {op} {elem2}",
        ],
    },

    "french": {
        "name": "French",
        "code": "fr",
        "worlds": {
            "A": ["royaume spirituel", "plan spirituel", "dimension spirituelle"],
            "B": ["royaume mental", "plan mental", "dimension mentale"],
            "C": ["royaume emotionnel", "plan emotionnel", "dimension emotionnelle"],
            "D": ["royaume physique", "plan materiel", "dimension physique"],
        },
        "positions": {
            "1": ["conscience", "esprit", "awareness"],
            "2": ["polarite positive", "force attractive", "expansion"],
            "3": ["polarite negative", "force repulsive", "contraction"],
            "4": ["vibration", "frequence", "oscillation"],
            "5": ["principe feminin", "receptivite", "entree"],
            "6": ["principe masculin", "projection", "sortie"],
            "7": ["rythme", "cycle", "periodicite"],
            "8": ["cause", "origine", "source"],
            "9": ["effet", "resultat", "manifestation"],
            "10": ["totalite", "integrite", "tout"],
        },
        "operators": {
            "+T": ["accumule temporellement avec", "croit avec le temps avec"],
            "-T": ["diminue temporellement de", "reduit avec le temps de"],
            "*T": ["amplifie temporellement avec", "resonne avec"],
            "/T": ["attenue temporellement a travers", "disperse avec le temps"],
            "+": ["combine avec", "fusionne avec"],
            "-": ["separe de", "differencie de"],
            "->": ["coule vers", "se transforme en"],
            "<-": ["derive de", "recoit de"],
            "o": ["cycle avec", "circule a travers"],
        },
        "templates": [
            "Le {elem1} {op} le {elem2}",
            "{elem1} {op} {elem2} dans ce travail",
            "Cette equation montre {elem1} {op} {elem2}",
        ],
    },

    "portuguese": {
        "name": "Portuguese",
        "code": "pt",
        "worlds": {
            "A": ["reino espiritual", "plano espiritual", "dimensao espiritual"],
            "B": ["reino mental", "plano mental", "dimensao mental"],
            "C": ["reino emocional", "plano emocional", "dimensao emocional"],
            "D": ["reino fisico", "plano material", "dimensao fisica"],
        },
        "positions": {
            "1": ["consciencia", "mente", "percepcao"],
            "2": ["polaridade positiva", "forca atrativa", "expansao"],
            "3": ["polaridade negativa", "forca repulsiva", "contracao"],
            "4": ["vibracao", "frequencia", "oscilacao"],
            "5": ["principio feminino", "receptividade", "entrada"],
            "6": ["principio masculino", "projecao", "saida"],
            "7": ["ritmo", "ciclo", "periodicidade"],
            "8": ["causa", "origem", "fonte"],
            "9": ["efeito", "resultado", "manifestacao"],
            "10": ["totalidade", "integridade", "tudo"],
        },
        "operators": {
            "+T": ["acumula temporalmente com", "cresce com o tempo com"],
            "-T": ["diminui temporalmente de", "reduz com o tempo de"],
            "*T": ["amplifica temporalmente com", "ressoa com"],
            "/T": ["atenua temporalmente atraves de", "dispersa com o tempo"],
            "+": ["combina com", "une-se com"],
            "-": ["separa de", "diferencia de"],
            "->": ["flui para", "transforma-se em"],
            "<-": ["deriva de", "recebe de"],
            "o": ["cicla com", "circula atraves de"],
        },
        "templates": [
            "O {elem1} {op} o {elem2}",
            "{elem1} {op} {elem2} neste trabalho",
            "Esta equacao mostra {elem1} {op} {elem2}",
        ],
    },

    "italian": {
        "name": "Italian",
        "code": "it",
        "worlds": {
            "A": ["regno spirituale", "piano spirituale", "dimensione spirituale"],
            "B": ["regno mentale", "piano mentale", "dimensione mentale"],
            "C": ["regno emotivo", "piano emotivo", "dimensione emotiva"],
            "D": ["regno fisico", "piano materiale", "dimensione fisica"],
        },
        "positions": {
            "1": ["coscienza", "mente", "consapevolezza"],
            "2": ["polarita positiva", "forza attrattiva", "espansione"],
            "3": ["polarita negativa", "forza repulsiva", "contrazione"],
            "4": ["vibrazione", "frequenza", "oscillazione"],
            "5": ["principio femminile", "ricettivita", "ingresso"],
            "6": ["principio maschile", "proiezione", "uscita"],
            "7": ["ritmo", "ciclo", "periodicita"],
            "8": ["causa", "origine", "fonte"],
            "9": ["effetto", "risultato", "manifestazione"],
            "10": ["totalita", "integrita", "tutto"],
        },
        "operators": {
            "+T": ["accumula temporalmente con", "cresce nel tempo con"],
            "-T": ["diminuisce temporalmente da", "riduce nel tempo da"],
            "*T": ["amplifica temporalmente con", "risuona con"],
            "/T": ["attenua temporalmente attraverso", "disperde nel tempo"],
            "+": ["combina con", "unisce con"],
            "-": ["separa da", "differenzia da"],
            "->": ["fluisce verso", "si trasforma in"],
            "<-": ["deriva da", "riceve da"],
            "o": ["cicla con", "circola attraverso"],
        },
        "templates": [
            "Il {elem1} {op} il {elem2}",
            "{elem1} {op} {elem2} in questo lavoro",
            "Questa equazione mostra {elem1} {op} {elem2}",
        ],
    },

    # =========================================================================
    # GERMANIC LANGUAGES
    # =========================================================================
    "german": {
        "name": "German",
        "code": "de",
        "worlds": {
            "A": ["geistiges Reich", "spirituelle Ebene", "geistige Dimension"],
            "B": ["mentales Reich", "mentale Ebene", "geistige Dimension"],
            "C": ["emotionales Reich", "emotionale Ebene", "gefuehlsmaessige Dimension"],
            "D": ["physisches Reich", "materielle Ebene", "koerperliche Dimension"],
        },
        "positions": {
            "1": ["Bewusstsein", "Geist", "Wahrnehmung"],
            "2": ["positive Polaritaet", "anziehende Kraft", "Ausdehnung"],
            "3": ["negative Polaritaet", "abstossende Kraft", "Zusammenziehung"],
            "4": ["Schwingung", "Frequenz", "Oszillation"],
            "5": ["weibliches Prinzip", "Empfaenglichkeit", "Eingang"],
            "6": ["maennliches Prinzip", "Projektion", "Ausgang"],
            "7": ["Rhythmus", "Zyklus", "Periodizitaet"],
            "8": ["Ursache", "Ursprung", "Quelle"],
            "9": ["Wirkung", "Ergebnis", "Manifestation"],
            "10": ["Gesamtheit", "Ganzheit", "Alles"],
        },
        "operators": {
            "+T": ["akkumuliert zeitlich mit", "waechst mit der Zeit mit"],
            "-T": ["nimmt zeitlich ab von", "reduziert mit der Zeit von"],
            "*T": ["verstaerkt zeitlich mit", "resoniert mit"],
            "/T": ["daempft zeitlich durch", "zerstreut mit der Zeit"],
            "+": ["verbindet mit", "vereinigt mit"],
            "-": ["trennt von", "differenziert von"],
            "->": ["fliesst zu", "transformiert sich in"],
            "<-": ["leitet ab von", "empfaengt von"],
            "o": ["zykliert mit", "zirkuliert durch"],
        },
        "templates": [
            "Das {elem1} {op} das {elem2}",
            "{elem1} {op} {elem2} in dieser Arbeit",
            "Diese Gleichung zeigt {elem1} {op} {elem2}",
        ],
    },

    "dutch": {
        "name": "Dutch",
        "code": "nl",
        "worlds": {
            "A": ["spiritueel rijk", "geestelijk vlak", "spirituele dimensie"],
            "B": ["mentaal rijk", "mentaal vlak", "mentale dimensie"],
            "C": ["emotioneel rijk", "emotioneel vlak", "emotionele dimensie"],
            "D": ["fysiek rijk", "materieel vlak", "fysieke dimensie"],
        },
        "positions": {
            "1": ["bewustzijn", "geest", "waarneming"],
            "2": ["positieve polariteit", "aantrekkende kracht", "expansie"],
            "3": ["negatieve polariteit", "afstotende kracht", "contractie"],
            "4": ["trilling", "frequentie", "oscillatie"],
            "5": ["vrouwelijk principe", "ontvankelijkheid", "ingang"],
            "6": ["mannelijk principe", "projectie", "uitgang"],
            "7": ["ritme", "cyclus", "periodiciteit"],
            "8": ["oorzaak", "oorsprong", "bron"],
            "9": ["effect", "resultaat", "manifestatie"],
            "10": ["totaliteit", "heelheid", "alles"],
        },
        "operators": {
            "+T": ["accumuleert tijdelijk met", "groeit met de tijd met"],
            "-T": ["neemt tijdelijk af van", "vermindert met de tijd van"],
            "*T": ["versterkt tijdelijk met", "resoneert met"],
            "/T": ["dempt tijdelijk door", "verspreidt met de tijd"],
            "+": ["combineert met", "verenigt met"],
            "-": ["scheidt van", "differentieert van"],
            "->": ["stroomt naar", "transformeert in"],
            "<-": ["is afgeleid van", "ontvangt van"],
            "o": ["cycleert met", "circuleert door"],
        },
        "templates": [
            "De {elem1} {op} de {elem2}",
            "{elem1} {op} {elem2} in dit werk",
            "Deze vergelijking toont {elem1} {op} {elem2}",
        ],
    },

    "swedish": {
        "name": "Swedish",
        "code": "sv",
        "worlds": {
            "A": ["andligt rike", "andligt plan", "andlig dimension"],
            "B": ["mentalt rike", "mentalt plan", "mental dimension"],
            "C": ["emotionellt rike", "emotionellt plan", "emotionell dimension"],
            "D": ["fysiskt rike", "materiellt plan", "fysisk dimension"],
        },
        "positions": {
            "1": ["medvetande", "sinne", "uppfattning"],
            "2": ["positiv polaritet", "attraherande kraft", "expansion"],
            "3": ["negativ polaritet", "frastotande kraft", "kontraktion"],
            "4": ["vibration", "frekvens", "oscillation"],
            "5": ["kvinnlig princip", "mottaglighet", "ingang"],
            "6": ["manlig princip", "projektion", "utgang"],
            "7": ["rytm", "cykel", "periodicitet"],
            "8": ["orsak", "ursprung", "kalla"],
            "9": ["effekt", "resultat", "manifestation"],
            "10": ["totalitet", "helhet", "allt"],
        },
        "operators": {
            "+T": ["ackumulerar temporart med", "vaxer over tid med"],
            "-T": ["minskar temporart fran", "reducerar over tid fran"],
            "*T": ["forstarker temporart med", "resonerar med"],
            "/T": ["dampar temporart genom", "sprider over tid"],
            "+": ["kombinerar med", "forenar med"],
            "-": ["separerar fran", "differentierar fran"],
            "->": ["floder till", "transformeras till"],
            "<-": ["harror fran", "mottar fran"],
            "o": ["cyklar med", "cirkulerar genom"],
        },
        "templates": [
            "Den {elem1} {op} den {elem2}",
            "{elem1} {op} {elem2} i detta arbete",
            "Denna ekvation visar {elem1} {op} {elem2}",
        ],
    },

    # =========================================================================
    # SLAVIC LANGUAGES
    # =========================================================================
    "russian": {
        "name": "Russian (Romanized)",
        "code": "ru",
        "worlds": {
            "A": ["dukhovnoe tsarstvo", "dukhovnyy plan", "dukhovnoe izmerenie"],
            "B": ["mental'noe tsarstvo", "mental'nyy plan", "mental'noe izmerenie"],
            "C": ["emotsional'noe tsarstvo", "emotsional'nyy plan", "emotsional'noe izmerenie"],
            "D": ["fizicheskoe tsarstvo", "material'nyy plan", "fizicheskoe izmerenie"],
        },
        "positions": {
            "1": ["soznanie", "razum", "vospriyatie"],
            "2": ["polozhitel'naya polyarnost'", "prityagayushchaya sila", "rasshirenie"],
            "3": ["otritsatel'naya polyarnost'", "ottalkivayushchaya sila", "szhatie"],
            "4": ["vibratsiya", "chastota", "ostsillyatsiya"],
            "5": ["zhenskiy printsip", "vospriyativost'", "vkhod"],
            "6": ["muzhskoy printsip", "proyektsiya", "vykhod"],
            "7": ["ritm", "tsikl", "periodichnost'"],
            "8": ["prichina", "proiskhozhdenie", "istochnik"],
            "9": ["effekt", "rezul'tat", "proyavlenie"],
            "10": ["tselostnost'", "polnota", "vsyo"],
        },
        "operators": {
            "+T": ["nakhodit vremenno s", "rastyot so vremenem s"],
            "-T": ["umen'shaetsya vremenno ot", "sokrashchayetsya so vremenem ot"],
            "*T": ["usilivayet vremenno s", "rezoniruyet s"],
            "/T": ["oslablyayet vremenno cherez", "rasseivayetsya so vremenem"],
            "+": ["ob\"yedinyayetsya s", "soedinyayetsya s"],
            "-": ["otdelyayetsya ot", "razlichayetsya ot"],
            "->": ["techet k", "prevrashchayetsya v"],
            "<-": ["proizkhodit ot", "poluchayeт ot"],
            "o": ["tsikliruyet s", "tsirkuliruyet cherez"],
        },
        "templates": [
            "{elem1} {op} {elem2}",
            "{elem1} {op} {elem2} v etoy rabote",
            "Eto uravnenie pokazyvayet {elem1} {op} {elem2}",
        ],
    },

    "polish": {
        "name": "Polish",
        "code": "pl",
        "worlds": {
            "A": ["krolestwo duchowe", "plan duchowy", "wymiar duchowy"],
            "B": ["krolestwo mentalne", "plan mentalny", "wymiar mentalny"],
            "C": ["krolestwo emocjonalne", "plan emocjonalny", "wymiar emocjonalny"],
            "D": ["krolestwo fizyczne", "plan materialny", "wymiar fizyczny"],
        },
        "positions": {
            "1": ["swiadomosc", "umysl", "percepcja"],
            "2": ["biegunowowsc pozytywna", "sila przyciagajaca", "ekspansja"],
            "3": ["biegunowowsc negatywna", "sila odpychajaca", "kontrakcja"],
            "4": ["wibracja", "czestotliwosc", "oscylacja"],
            "5": ["zasada zenska", "receptywnosc", "wejscie"],
            "6": ["zasada meska", "projekcja", "wyjscie"],
            "7": ["rytm", "cykl", "okresowowsc"],
            "8": ["przyczyna", "poczatek", "zrodlo"],
            "9": ["efekt", "rezultat", "manifestacja"],
            "10": ["calosc", "pelnia", "wszystko"],
        },
        "operators": {
            "+T": ["akumuluje czasowo z", "rosnie z czasem z"],
            "-T": ["zmniejsza sie czasowo od", "redukuje z czasem od"],
            "*T": ["wzmacnia czasowo z", "rezonuje z"],
            "/T": ["tlumi czasowo przez", "rozprasza z czasem"],
            "+": ["laczy sie z", "jednoczy sie z"],
            "-": ["oddziela sie od", "roznicuje od"],
            "->": ["plynie do", "przeksztalca sie w"],
            "<-": ["pochodzi od", "otrzymuje od"],
            "o": ["cykluje z", "cyrkuluje przez"],
        },
        "templates": [
            "{elem1} {op} {elem2}",
            "{elem1} {op} {elem2} w tej pracy",
            "To rownanie pokazuje {elem1} {op} {elem2}",
        ],
    },

    # =========================================================================
    # ASIAN LANGUAGES (Romanized)
    # =========================================================================
    "japanese": {
        "name": "Japanese (Romanized)",
        "code": "ja",
        "worlds": {
            "A": ["reikai", "seishin no ryouiki", "tamashii no jigen"],
            "B": ["seishin ryouiki", "kokoro no ryouiki", "chishiki no jigen"],
            "C": ["kanjou ryouiki", "kimochi no ryouiki", "joucho no jigen"],
            "D": ["busshitsu ryouiki", "nikutai no ryouiki", "butsurigaku no jigen"],
        },
        "positions": {
            "1": ["ishiki", "kokoro", "ninshiki"],
            "2": ["youseikyoku", "hikinetsuke no chikara", "kakuchou"],
            "3": ["inseikyoku", "hanpatsu no chikara", "shuushuku"],
            "4": ["shindou", "shuuhasuu", "yuremeki"],
            "5": ["josei genri", "uketoreru", "nyuuryoku"],
            "6": ["dansei genri", "tousya", "shutsuryoku"],
            "7": ["rizumu", "saikuru", "shuuki"],
            "8": ["gen'in", "kigen", "minamoto"],
            "9": ["kouka", "kekka", "hyougen"],
            "10": ["zentai", "kanzen", "subete"],
        },
        "operators": {
            "+T": ["jikan teki ni chikuseki suru", "jikan to tomo ni seicho suru"],
            "-T": ["jikan teki ni genshou suru", "jikan to tomo ni herasu"],
            "*T": ["jikan teki ni zoufuku suru", "kyoumei suru"],
            "/T": ["jikan teki ni gensui suru", "jikan to tomo ni bunsan suru"],
            "+": ["ketsugou suru", "gappei suru"],
            "-": ["bunri suru", "kubetsu suru"],
            "->": ["nagareru", "henka suru"],
            "<-": ["yurai suru", "ukeru"],
            "o": ["junkan suru", "meguru"],
        },
        "templates": [
            "{elem1} wa {elem2} ni {op}",
            "Kono hataraki de {elem1} ga {elem2} ni {op}",
            "Kono housiki wa {elem1} ga {elem2} ni {op} koto wo shimesu",
        ],
    },

    "chinese": {
        "name": "Chinese (Pinyin)",
        "code": "zh",
        "worlds": {
            "A": ["ling xing ling yu", "jing shen ceng mian", "ling hun wei du"],
            "B": ["xin li ling yu", "si wei ceng mian", "jing shen wei du"],
            "C": ["qing gan ling yu", "qing xu ceng mian", "gan qing wei du"],
            "D": ["wu zhi ling yu", "wu li ceng mian", "shen ti wei du"],
        },
        "positions": {
            "1": ["yi shi", "xin zhi", "ren zhi"],
            "2": ["zheng ji xing", "xi yin li", "kuo zhang"],
            "3": ["fu ji xing", "pai chi li", "shou suo"],
            "4": ["zhen dong", "pin lu", "bo dong"],
            "5": ["yin xing yuan li", "jie shou xing", "shu ru"],
            "6": ["yang xing yuan li", "tou she", "shu chu"],
            "7": ["jie zou", "zhou qi", "zhou qi xing"],
            "8": ["yuan yin", "qi yuan", "lai yuan"],
            "9": ["xiao guo", "jie guo", "biao xian"],
            "10": ["zheng ti", "wan zheng", "quan bu"],
        },
        "operators": {
            "+T": ["sui shi jian lei ji", "sui shi jian zeng zhang"],
            "-T": ["sui shi jian jian shao", "sui shi jian xiao jian"],
            "*T": ["sui shi jian fang da", "gong zhen"],
            "/T": ["sui shi jian jian ruo", "sui shi jian fen san"],
            "+": ["jie he", "he bing"],
            "-": ["fen li", "qu fen"],
            "->": ["liu xiang", "zhuan hua wei"],
            "<-": ["lai zi", "jie shou zi"],
            "o": ["xun huan", "liu tong"],
        },
        "templates": [
            "{elem1} {op} {elem2}",
            "Zai zhe ge gong zuo zhong {elem1} {op} {elem2}",
            "Zhe ge fang cheng shi biao shi {elem1} {op} {elem2}",
        ],
    },

    "korean": {
        "name": "Korean (Romanized)",
        "code": "ko",
        "worlds": {
            "A": ["yeongjeok yeongeyeok", "jeongsin pyeong myeon", "yeongjeok chawon"],
            "B": ["jeongsin yeongeyeok", "sasang pyeong myeon", "jeongsin chawon"],
            "C": ["gamjeong yeongeyeok", "gamjeong pyeong myeon", "gamjeong chawon"],
            "D": ["mulli yeongeyeok", "mulil pyeong myeon", "yukche chawon"],
        },
        "positions": {
            "1": ["uisik", "maeum", "insik"],
            "2": ["yanggeuseong", "kkeullineun him", "hwakjang"],
            "3": ["eumgeuseong", "milchyoneneun him", "suchuk"],
            "4": ["jindong", "juphasu", "jindong"],
            "5": ["yeoseong wonri", "suyong seong", "ibryeok"],
            "6": ["namseong wonri", "tunyeong", "chulryeok"],
            "7": ["rideum", "jugi", "jukiseong"],
            "8": ["wonin", "giwon", "woncheon"],
            "9": ["hyogwa", "gyeolgwa", "balhyeon"],
            "10": ["jeonche", "wanjeon", "modeun geos"],
        },
        "operators": {
            "+T": ["sigane dara chukjeok", "sigane dara seongjang"],
            "-T": ["sigane dara gamso", "sigane dara chukso"],
            "*T": ["sigane dara jeunghok", "gongmyeong"],
            "/T": ["sigane dara gamso", "sigane dara bunsan"],
            "+": ["gyeolhap", "hapchyeo"],
            "-": ["bunni", "gubun"],
            "->": ["heulleo", "byeoni"],
            "<-": ["bisrot", "badaneun"],
            "o": ["sunhwan", "dolda"],
        },
        "templates": [
            "{elem1}eun {elem2}e {op}",
            "I jageop eseo {elem1}i {elem2}e {op}",
            "I bangjeongsik eun {elem1}i {elem2}e {op} geos eul boyeojunda",
        ],
    },

    "hindi": {
        "name": "Hindi (Romanized)",
        "code": "hi",
        "worlds": {
            "A": ["aadhyaatmik kshetra", "aadhyaatmik sthaan", "aatmik aayaam"],
            "B": ["maanasik kshetra", "maanasik sthaan", "maanasik aayaam"],
            "C": ["bhaavnaatmak kshetra", "bhaavnaatmak sthaan", "bhaavnaatmak aayaam"],
            "D": ["bhautik kshetra", "bhautik sthaan", "shaareerik aayaam"],
        },
        "positions": {
            "1": ["chetna", "man", "jagrukta"],
            "2": ["dhanaatmak dhruveeyata", "aakrshan bal", "vistaar"],
            "3": ["rinaaatmak dhruveeyata", "praatikrshan bal", "sankochan"],
            "4": ["kampan", "aavritti", "dolansheelta"],
            "5": ["streeling siddhaant", "graahyata", "pravesh"],
            "6": ["pulling siddhaant", "prakshepan", "nishkrshan"],
            "7": ["taal", "chakr", "aavartita"],
            "8": ["kaaran", "udgam", "shrota"],
            "9": ["parinam", "natija", "abhivyakti"],
            "10": ["sampoornata", "akhandta", "sab kuch"],
        },
        "operators": {
            "+T": ["samay ke saath sanchit", "samay ke saath badhta"],
            "-T": ["samay ke saath kam hota", "samay ke saath ghatta"],
            "*T": ["samay ke saath prasarit", "anugunjit hota"],
            "/T": ["samay ke saath kam", "samay ke saath bikharata"],
            "+": ["jodta hai", "milata hai"],
            "-": ["alag karta hai", "vibhajit karta hai"],
            "->": ["bahta hai", "roop badalta hai"],
            "<-": ["nikalti hai", "prapt karta hai"],
            "o": ["chakr karti hai", "ghoomti hai"],
        },
        "templates": [
            "{elem1} {elem2} ke saath {op}",
            "Is kaarya mein {elem1} {elem2} ke saath {op}",
            "Yah samikaran dikhata hai ki {elem1} {elem2} ke saath {op}",
        ],
    },

    # =========================================================================
    # AFRICAN LANGUAGES
    # =========================================================================
    "swahili": {
        "name": "Swahili",
        "code": "sw",
        "worlds": {
            "A": ["ulimwengu wa kiroho", "ngazi ya kiroho", "mwelekeo wa kiroho"],
            "B": ["ulimwengu wa akili", "ngazi ya akili", "mwelekeo wa kiakili"],
            "C": ["ulimwengu wa hisia", "ngazi ya hisia", "mwelekeo wa kihisia"],
            "D": ["ulimwengu wa kimwili", "ngazi ya kimwili", "mwelekeo wa kimwili"],
        },
        "positions": {
            "1": ["ufahamu", "akili", "utambuzi"],
            "2": ["polarity chanya", "nguvu ya kuvutia", "upanuzi"],
            "3": ["polarity hasi", "nguvu ya kusukuma", "kupungua"],
            "4": ["mtetemo", "masafa", "kusumbua"],
            "5": ["kanuni ya kike", "upokezi", "kuingia"],
            "6": ["kanuni ya kiume", "kutoa", "kutoka"],
            "7": ["mdundo", "mzunguko", "upimaji wa mara"],
            "8": ["sababu", "asili", "chanzo"],
            "9": ["athari", "matokeo", "udhihirisho"],
            "10": ["jumla", "ukamilifu", "yote"],
        },
        "operators": {
            "+T": ["hujilimbikiza na wakati na", "hukua na wakati na"],
            "-T": ["hupungua na wakati kutoka", "hupunguza na wakati kutoka"],
            "*T": ["huongeza na wakati na", "hurudia na"],
            "/T": ["hupunguza na wakati kupitia", "hutawanya na wakati"],
            "+": ["huungana na", "huchanganya na"],
            "-": ["hutenganisha na", "hutofautisha na"],
            "->": ["inapita kwa", "inabadilika kuwa"],
            "<-": ["inatokana na", "inapokea kutoka"],
            "o": ["inazunguka na", "inasambaa kupitia"],
        },
        "templates": [
            "{elem1} {op} {elem2}",
            "Katika kazi hii {elem1} {op} {elem2}",
            "Mlinganyo huu unaonyesha {elem1} {op} {elem2}",
        ],
    },

    # =========================================================================
    # SEMITIC LANGUAGES (Romanized)
    # =========================================================================
    "arabic": {
        "name": "Arabic (Romanized)",
        "code": "ar",
        "worlds": {
            "A": ["al-mamlaka al-ruhiyya", "al-mustawa al-ruhi", "al-buad al-ruhi"],
            "B": ["al-mamlaka al-aqliyya", "al-mustawa al-aqli", "al-buad al-aqli"],
            "C": ["al-mamlaka al-atifiyya", "al-mustawa al-atifi", "al-buad al-atifi"],
            "D": ["al-mamlaka al-jismiyya", "al-mustawa al-maddi", "al-buad al-jasadi"],
        },
        "positions": {
            "1": ["al-waay", "al-aql", "al-idraak"],
            "2": ["al-qutbiyya al-ijabiyya", "quwwat al-jadhb", "al-tawassua"],
            "3": ["al-qutbiyya al-salbiyya", "quwwat al-tanafur", "al-taqallus"],
            "4": ["al-ihtizaz", "al-taradud", "al-tadhabdhub"],
            "5": ["al-mabdaa al-unthawi", "al-istiqbal", "al-dukhul"],
            "6": ["al-mabdaa al-dhakari", "al-isqat", "al-khuruj"],
            "7": ["al-iqaa", "al-dawra", "al-tawatur"],
            "8": ["al-sabab", "al-asl", "al-masdar"],
            "9": ["al-athar", "al-natija", "al-tajalli"],
            "10": ["al-kulliyya", "al-tamam", "al-kull"],
        },
        "operators": {
            "+T": ["yatarakam maa al-waqt maa", "yanmu maa al-waqt maa"],
            "-T": ["yatanaqas maa al-waqt min", "yanqus maa al-waqt min"],
            "*T": ["yudaakhim maa al-waqt maa", "yataradad maa"],
            "/T": ["yukhafit maa al-waqt abr", "yatabaddad maa al-waqt"],
            "+": ["yandamij maa", "yattahid maa"],
            "-": ["yanfasil an", "yatamayyaz an"],
            "->": ["yatadaffaq ila", "yatahawwal ila"],
            "<-": ["yashtaqq min", "yatalaqqa min"],
            "o": ["yadur maa", "yatajawwal abr"],
        },
        "templates": [
            "{elem1} {op} {elem2}",
            "Fi hadha al-amal {elem1} {op} {elem2}",
            "Hadhihi al-muadala tudhir {elem1} {op} {elem2}",
        ],
    },
}

# =============================================================================
# ENGLISH VOCABULARY (for translations)
# =============================================================================

ENGLISH_VOCAB = {
    "worlds": {
        "A": ["spiritual realm", "transcendent plane", "spiritual dimension"],
        "B": ["mental realm", "cognitive plane", "mental dimension"],
        "C": ["emotional realm", "affective plane", "emotional dimension"],
        "D": ["physical realm", "material plane", "physical dimension"],
    },
    "positions": {
        "1": ["consciousness", "awareness", "mind"],
        "2": ["positive polarity", "attractive force", "expansion"],
        "3": ["negative polarity", "repulsive force", "contraction"],
        "4": ["vibration", "frequency", "oscillation"],
        "5": ["feminine principle", "receptivity", "intake"],
        "6": ["masculine principle", "projection", "output"],
        "7": ["rhythm", "cycle", "periodicity"],
        "8": ["cause", "origin", "source"],
        "9": ["effect", "result", "manifestation"],
        "10": ["totality", "wholeness", "integration"],
    },
    "operators": {
        "+T": ["accumulates temporally with", "builds over time with"],
        "-T": ["diminishes temporally from", "reduces over time from"],
        "*T": ["amplifies temporally with", "resonates over time with"],
        "/T": ["attenuates temporally through", "disperses over time through"],
        "+": ["combines with", "merges with"],
        "-": ["separates from", "differentiates from"],
        "->": ["flows to", "transforms into"],
        "<-": ["derives from", "receives from"],
        "o": ["cycles with", "circulates through"],
    },
    "templates": [
        "The {elem1} {op} the {elem2}",
        "{elem1} {op} {elem2} in this working",
        "This equation shows {elem1} {op} {elem2}",
    ],
}

# =============================================================================
# GENERATORS
# =============================================================================

def generate_element() -> str:
    """Generate random TKS element code."""
    world = random.choice(["A", "B", "C", "D"])
    pos = random.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"])
    elem = f"{world}{pos}"

    if random.random() < 0.2:
        sense = random.randint(1, 10)
        elem += f"^{sense}"

    if random.random() < 0.1:
        found = random.randint(1, 7)
        elem += f"_d{found}"

    return elem


def generate_chain(length: int = None) -> str:
    """Generate random TKS operator chain."""
    if length is None:
        length = random.randint(2, 4)

    elements = [generate_element() for _ in range(length)]
    ops = [random.choice(OPERATORS) for _ in range(length - 1)]

    chain = elements[0]
    for i, op in enumerate(ops):
        chain += f" {op} {elements[i + 1]}"

    return chain


def get_element_desc(code: str, vocab: dict) -> str:
    """Get element description using vocabulary."""
    world = code[0]
    rest = code[1:]
    if rest.startswith("10"):
        pos = "10"
    else:
        pos = rest[0] if rest else "1"

    world_desc = random.choice(vocab["worlds"].get(world, [WORLDS[world]]))
    pos_desc = random.choice(vocab["positions"].get(pos, [POSITIONS[pos]]))

    return f"{pos_desc} in {world_desc}"


def generate_language_sample(lang_key: str, lang: dict) -> Dict:
    """Generate one sample in target language."""
    chain = generate_chain()
    tokens = chain.split()
    elems = [t for t in tokens if t and t[0] in "ABCD"]
    ops = [t for t in tokens if t in OPERATORS]

    if len(elems) >= 2 and ops:
        elem1_desc = get_element_desc(elems[0], lang)
        elem2_desc = get_element_desc(elems[1], lang)
        op_desc = random.choice(lang["operators"].get(ops[0], [ops[0]]))

        template = random.choice(lang["templates"])
        story = template.format(elem1=elem1_desc, elem2=elem2_desc, op=op_desc)

        return {
            "story": story,
            "equation": chain,
            "source": f"lang_{lang_key}",
            "language": lang["name"]
        }

    return {"story": f"Working with {chain}", "source": f"lang_{lang_key}"}


def generate_translation_sample(chain: str, lang_key: str, lang: dict) -> Dict:
    """Generate translation pair between English and target language."""
    tokens = chain.split()
    elems = [t for t in tokens if t and t[0] in "ABCD"]
    ops = [t for t in tokens if t in OPERATORS]

    if len(elems) >= 2 and ops:
        # English version
        elem1_eng = get_element_desc(elems[0], ENGLISH_VOCAB)
        elem2_eng = get_element_desc(elems[1], ENGLISH_VOCAB)
        op_eng = random.choice(ENGLISH_VOCAB["operators"].get(ops[0], [ops[0]]))
        template_eng = random.choice(ENGLISH_VOCAB["templates"])
        english = template_eng.format(elem1=elem1_eng, elem2=elem2_eng, op=op_eng)

        # Target language version
        elem1_lang = get_element_desc(elems[0], lang)
        elem2_lang = get_element_desc(elems[1], lang)
        op_lang = random.choice(lang["operators"].get(ops[0], [ops[0]]))
        template_lang = random.choice(lang["templates"])
        target = template_lang.format(elem1=elem1_lang, elem2=elem2_lang, op=op_lang)

        # Randomly choose direction
        if random.random() < 0.5:
            return {
                "input": f"Translate to {lang['name']}: {english}",
                "output": target,
                "equation": chain,
                "source": f"translation_{lang_key}"
            }
        else:
            return {
                "input": f"Translate to English: {target}",
                "output": english,
                "equation": chain,
                "source": f"translation_{lang_key}"
            }

    return {"story": f"Working with {chain}", "source": f"translation_{lang_key}"}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate multilingual TKS samples")
    parser.add_argument("--output", default="output/train_multilingual.jsonl")
    parser.add_argument("--per-language", type=int, default=100000)
    parser.add_argument("--translations-per-language", type=int, default=50000)
    args = parser.parse_args()

    lang_keys = list(LANGUAGES.keys())

    print("=" * 60)
    print("TKS MULTILINGUAL GENERATOR")
    print("=" * 60)
    print(f"\nLanguages: {len(lang_keys)}")
    for i, key in enumerate(lang_keys, 1):
        print(f"  {i}. {LANGUAGES[key]['name']}")

    print(f"\nTarget samples:")
    print(f"  Language samples: {len(lang_keys)} x {args.per_language:,} = {len(lang_keys) * args.per_language:,}")
    print(f"  Translation pairs: {len(lang_keys)} x {args.translations_per_language:,} = {len(lang_keys) * args.translations_per_language:,}")

    total_target = (len(lang_keys) * args.per_language +
                   len(lang_keys) * args.translations_per_language)
    print(f"  TOTAL TARGET: {total_target:,}")
    print()

    all_samples = []

    # 1. Generate language samples
    print("Generating language samples...")
    for lang_key in lang_keys:
        lang = LANGUAGES[lang_key]
        print(f"  {lang['name']}...", end=" ", flush=True)
        for _ in range(args.per_language):
            sample = generate_language_sample(lang_key, lang)
            all_samples.append(sample)
        print(f"{args.per_language:,}")

    # 2. Generate translation pairs
    print("\nGenerating translation pairs...")
    for lang_key in lang_keys:
        lang = LANGUAGES[lang_key]
        print(f"  English <-> {lang['name']}...", end=" ", flush=True)
        for _ in range(args.translations_per_language):
            chain = generate_chain()
            sample = generate_translation_sample(chain, lang_key, lang)
            all_samples.append(sample)
        print(f"{args.translations_per_language:,}")

    # Shuffle and deduplicate
    print("\nShuffling and deduplicating...")
    random.shuffle(all_samples)

    seen = set()
    final = []
    for item in all_samples:
        key = item.get("story", item.get("input", ""))
        if key and key not in seen:
            seen.add(key)
            final.append(item)

    print(f"\nTotal generated: {len(all_samples):,}")
    print(f"Unique samples: {len(final):,}")
    print(f"Uniqueness: {len(final)/len(all_samples)*100:.1f}%")

    # Save
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nSaved to: {args.output}")

    # Distribution
    print("\nDistribution by source (top 20):")
    sources = {}
    for item in final:
        src = item.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    for src, cnt in sorted(sources.items(), key=lambda x: -x[1])[:20]:
        pct = cnt / len(final) * 100
        print(f"  {src}: {cnt:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
