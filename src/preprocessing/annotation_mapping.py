"""
Mapovanie anotácií MIT-BIH databázy do štandardizovaných AAMI tried
-------------------------------------------------------------------
* MIT-BIH Arrhythmia databáza obsahuje viac ako 15 rôznych typov úderov.
* Táto mapa prevádza každý symbol na jednu z 5 hlavných tried podľa odporúčania AAMI: N, S, V, F, Q.
* Funkcia `map_mitbih_annotation()` zabezpečuje konzistentnú klasifikáciu aj pre zriedkavé alebo chybné anotácie.
* Neznáme alebo chýbajúce symboly sú automaticky zaradené do triedy Q a zaznamenané do logu.
* Modul je kľúčový pre prípravu trénovacích a testovacích dát pre klasifikačné modely.
"""
#https://www.researchgate.net/figure/Heartbeat-annotations-in-MIT-BIH-dataset-according-to-AAMI-EC-57-The-consolidated_fig1_372894714
import logging


ANNOTATION_MAP = {
    # Skupina N (Normálny úder)
    'N': 'N',  # Normálny úder
    '.': 'N',  # Normálny úder (niekedy označovaný bodkou)
    'L': 'N',  # Úder s blokom ľavého Tawarovho ramienka
    'R': 'N',  # Úder s blokom pravého Tawarovho ramienka
    'e': 'N',  # Atriálny únikový úder
    'j': 'N',  # Nódový (junkčný) únikový úder

    # Skupina S (Supraventrikulárne ektopické údery)
    'A': 'S',  # Predčasný atriálny úder
    'a': 'S',  # Aberantný predčasný atriálny úder
    'J': 'S',  # Predčasný junkčný úder
    'S': 'S',  # Supraventrikulárny predčasný úder

    # Skupina V (Ventrikulárne ektopické údery)
    'V': 'V',  # Predčasná ventrikulárna kontrakcia
    'E': 'V',  # Ventrikulárny únikový úder

    # Skupina F (Fúzne údery)
    'F': 'F',  # Fúzia ventrikulárneho a normálneho úderu

    # Skupina Q (Neznáme alebo neklasifikovateľné údery)
    '/': 'Q',   # Paced beat
    'f': 'Q',   # Fusion of paced and normal beat
    'Q': 'Q',   # Unknown/unclassified beat
    '?': 'Q',   # Questionable beat
    '|': 'Q'    # Isolated QRS-like artifact
}


def map_mitbih_annotation(symbol: str) -> str:
    """
       Funkcia mapuje MIT-BIH anotáciu (napr. 'N', 'V', 'J', '?')
       na jednu zo štandardizovaných tried AAMI: N, S, V, F, Q.

       V prípade, že symbol nie je známy, zaradí ho do triedy 'Q' (neznáme),
       a zároveň vypíše varovanie do logu.
       """
    if symbol not in ANNOTATION_MAP:
        logging.warning(f"Neznámy symbol: '{symbol}', zaradený do triedy Q")
        return 'Q'
    return ANNOTATION_MAP[symbol]
