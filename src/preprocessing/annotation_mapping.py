# preprocessing/annotation_mapping.py
# Definovanie mapovania jednotlivých MIT-BIH anotácií
# na požadované 5 triedy: N, S, V, F, Q
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
    'P': 'Q',  # Stimulovaný (paced) úder
    '/': 'Q',  # Stimulovaný úder (označený lomkou)
    'f': 'Q',  # Fúzia stimulovaného a normálneho úderu
    'U': 'Q'   # Neklasifikovateľný úder
}


def map_mitbih_annotation(symbol: str) -> str:
    """
    Funkcia premapuje jeden MIT-BIH symbol na jednu z piatich tried:
    N (normálny), S (supraventrikulárny), V (ventrikulárny),
    F (fúzny), Q (neznámy).
    """
    return ANNOTATION_MAP.get(symbol, 'Q')  # Ak symbol nie je v mape, vráti 'Q' ako neznámu triedu