# # DEMO.PY: Named Entity example 
# # import spacy a NLP lib for do NLP
# import spacy

# # Load the big English model of NLP
# # for load the module: python -m spacy download en_core_web_lg or en_core_web_sm
# nlp = spacy.load('en_core_web_lg')
# # nlp = spacy.load("en_core_web_sm")

# # The text is we want to examine
# text = """London is the capital and most populous city of England and 
# the United Kingdom.  Standing on the River Thames in the south east 
# of the island of Great Britain, London has been a major settlement 
# for two millennia. It was founded by the Romans, who named it Londinium.
# """

# # parse the text with spaCy. This runs entire pipline.
# doc = nlp(text)
# print(doc)

# # 'doc' now contains a parsed version of text, we can use it to do anything we want.
# # For ex. this will print out the named entities that were detected:
# for entity in doc.ents:
# 	print(f"{entity.text} ({entity.label_})")
# """I/O for en_core_web_sm
# London (GPE)
# England (GPE)
# the United Kingdom (GPE)
# the south east (LOC)
# Great Britain (GPE)
# London (GPE)
# two (CARDINAL)
# Romans (NORP)
# Londinium (ORG)"""

# """I/O for en_core_web_lg
# London (GPE)
# England (GPE)
# the United Kingdom (GPE)
# the River Thames (LOC)
# the south east 
#  (LOC)
# Great Britain (GPE)
# London (GPE)
# two millennia (DATE)
# Romans (NORP)
# Londinium (ORG)"""
# #and search differences between them

# # and try for both them too
# # spacy.displacy.serve(doc, style="ent")
# # spacy.displacy.render(doc, style="dep")

# PII_SCRUBBER.PY
# import spacy
# nlp = spacy.load('en_core_web_sm')

# # replace a token with REDACTED if it is a name.
# def replace_name_with_placeholder(token):
# 	if token.ent_iob != 0 and token.ent_type == "PERSON":
# 		return "[REDACTED] "
# 	else:
# 		return token.string
# # Loop thgough all the entities in a doc and check if they are names.
# def scrub(text):
# 	doc = nlp(text)
# 	for ent in doc.ents:
# 		ent.merge()
# 	tokens = map(replace_name_with_placeholder, doc)
# 	return "".join(tokens)
# s = """
# In 1950, Alan Turing published his famous article "Computing Machinery and Intelligence". In 1957, Noam Chomsky’s 
# Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.
# """
# print(scrub(s))
# I/O:
# In 1950, [REDACTED] published his famous article "Computing Machinery and Intelligence". In 1957, [REDACTED] 
# Syntactic Structures revolutionized Linguistics with 'universal grammar', a rule based system of syntactic structures.

# # FAC_EXTRACTION.PY :Coreference Resolution
# import spacy
# import textacy.extract as te
# nlp = spacy.load('en_core_web_lg')
# # The text we want to examine
# text = """London is the capital and most populous city of England and  the United Kingdom.  
# Standing on the River Thames in the south east of the island of Great Britain, 
# London has been a major settlement  for two millennia.  It was founded by the Romans, 
# who named it Londinium.
# """
# # or try on London text in wiki. 
# text1 = """ London is the capital and largest city of England and of the United Kingdom.[7][8] Standing on the River Thames in the south-east of England, at the head of its 50-mile (80 km) estuary leading to the North Sea, London has been a major settlement for two millennia. Londinium was founded by the Romans.[9] The City of London, London's ancient core − an area of just 1.12 square miles (2.9 km2) and colloquially known as the Square Mile − retains boundaries that closely follow its medieval limits.[10][11][12][13][14][note 1] The City of Westminster is also an Inner London borough holding city status. Greater London is governed by the Mayor of London and the London Assembly.[15][note 2][16]

# London is considered to be one of the world's most important global cities[17][18][19] and has been termed the world's most powerful,[20] most desirable,[21] most influential,[22] most visited,[23] most expensive,[24][25] innovative,[26] sustainable,[27] most investment friendly,[28] and most popular for work[29] city. London exerts a considerable impact upon the arts, commerce, education, entertainment, fashion, finance, healthcare, media, professional services, research and development, tourism and transportation.[30][31] London ranks 26th out of 300 major cities for economic performance.[32] It is one of the largest financial centres[33] and has either the fifth or the sixth largest metropolitan area GDP.[note 3][34][35][36][37][38] It is the most-visited city as measured by international arrivals[39] and has the busiest city airport system as measured by passenger traffic.[40] It is the leading investment destination,[41][42][43][44] hosting more international retailers[45][46] and ultra high-net-worth individuals[47][48] than any other city. London's universities form the largest concentration of higher education institutes in Europe,[49] and London is home to highly ranked institutions such as Imperial College London in natural and applied sciences, the London School of Economics in social sciences, and the comprehensive University College London and King's College London.[50][51][52] In 2012, London became the first city to have hosted three modern Summer Olympic Games.[53]

# London has a diverse range of people and cultures, and more than 300 languages are spoken in the region.[54] Its estimated mid-2018 municipal population (corresponding to Greater London) was 8,908,081,[4] the third most populous of any city in Europe[55] and accounts for 13.4% of the UK population.[56] London's urban area is the third most populous in Europe, after Moscow and Paris, with 9,787,426 inhabitants at the 2011 census.[57] The population within the London commuter belt is the most populous in Europe with 14,040,163 inhabitants in 2016.[note 4][3][58] London was the world's most populous city from c. 1831 to 1925.[59]

# London contains four World Heritage Sites: the Tower of London; Kew Gardens; the site comprising the Palace of Westminster, Westminster Abbey, and St Margaret's Church; and the historic settlement in Greenwich where the Royal Observatory, Greenwich defines the Prime Meridian (0° longitude) and Greenwich Mean Time.[60] Other landmarks include Buckingham Palace, the London Eye, Piccadilly Circus, St Paul's Cathedral, Tower Bridge, Trafalgar Square and The Shard. London has numerous museums, galleries, libraries and sporting events. These include the British Museum, National Gallery, Natural History Museum, Tate Modern, British Library and West End theatres.[61] The London Underground is the oldest underground railway network in the world."""

# doc = nlp(text)
# # Extract semi-structured statements
# statements = te.semistructured_statements(doc,"London")

# for statement in statements:
# 	subject, verb, fact = statement
# 	print(f" - {fact}")

# # KEY_TERMS.PY
# import spacy
# import textacy.extract
# nlp = spacy.load('en_core_web_lg')
# # The text we want to examine
# text = """London is [.. shorted for space ..]"""
# doc = nlp(text)
# # Extract noun chunks that appear
# noun_chunks = textacy.extract.noun_chunks(doc, min_freq=3)

# # Convert noun to lowercase
# noun_chunks = map(str, noun_chunks)
# noun_chunks = map(str.lower, noun_chunks)

# # print out any nouns that are at least 2 words long
# for noun_chunk in set(noun_chunks):
# 	if len(noun_chunk.split(" ")) > 1:
# 		print(noun_chunk)

# """
# I/O
# westminster abbey
# natural history museum
# west end
# east end
# st paul's cathedral
# royal albert hall
# london underground
# great fire
# british museum
# london eye.... etc ...."""

# source: I took help from 'NLP is fun' for this exercise