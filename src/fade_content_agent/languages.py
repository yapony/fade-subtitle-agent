"""Language-code normalization shared by prompt selection and source matching."""


def language_code(language: str|None) -> str|None:
    value=' '.join(str(language or '').strip().casefold().replace('_','-').split())
    if value in {'es','es-es','spa','spanish','español','espanol','西班牙语'} or value.startswith(('es-','spa-')):
        return 'es'
    if value in {'fr','fr-fr','fra','fre','french','français','francais','法语'} or value.startswith(('fr-','fra-','fre-')):
        return 'fr'
    return None
