from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,
    NewsNERTagger,
    
    PER,
    NamesExtractor,
    DatesExtractor,
    MoneyExtractor,
    AddrExtractor,

    Doc
)

text = "Динамика возобновления роста случаев коронавируса COVID-19 в мире является поводом для серьезной обеспокоенности – признал преcс-секретарь президента России Дмитрий Песков."

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)

names_extractor = NamesExtractor(morph_vocab)
dates_extractor = DatesExtractor(morph_vocab)
money_extractor = MoneyExtractor(morph_vocab)
addr_extractor = AddrExtractor(morph_vocab)

doc = Doc(text)
doc.segment(segmenter)
doc.tag_morph(morph_tagger)
doc.parse_syntax(syntax_parser)
doc.tag_ner(ner_tagger)

for token in doc.tokens:
    token.lemmatize(morph_vocab)
    
for token in doc.tokens:
    print(token.lemma)
    
#display({_.text: _.lemma for _ in doc.tokens})

for span in doc.spans:
    span.normalize(morph_vocab)
print(doc.spans)
    
#display({_.text: _.normal for _ in doc.spans})