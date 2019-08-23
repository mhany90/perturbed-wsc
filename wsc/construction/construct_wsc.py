"""
The city councilmen refused the demonstrators a permit because" "they"  "feared violence.
"                       "they"  "feared violence"               "The city councilmen"   "A"     "(Winograd 1972)"
                                                                                        "The demonstrators""
"""
import xml.etree.ElementTree as ET
path_to_wsc = '../data/wsc_data/WSCollection.xml'
wsc_file = open(path_to_wsc, 'r')


root = ET.parse(wsc_file).getroot()

example_list = []
skip_once = True

for child in root.iter():
    #print(child.tag.strip(), child.text)
    if child.tag.strip() == "txt1":
        txt1 = child.text.strip('"').replace('\n',' ').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "pron":
        pron = child.text.strip('"').replace('\n','').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "txt2":
        txt2 = child.text.strip('"').replace('\n',' ').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "quote2":
        quote = child.text.strip('"').replace('\n','').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "answer" and not answer1:
        answer1 = child.text.strip('"').replace('\n','').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "answer" and child.text != answer1:
        answer2 = child.text.strip('"').replace('\n','').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "correctAnswer":
        correctAnswer = child.text.strip('"').replace('\n','').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")
    if child.tag.strip() == "source":
        source = child.text.strip('"').replace('\n','').replace('.', ' . ').replace(',', ' , ').replace('toÂ','to').replace('CanopyHuntertropic', "Canopy Huntertropic")


    if child.tag.strip() == "schema" and not skip_once:
        txt1_len = len(txt1.split())

        #indices
        pron_index = txt1_len
        optionA_indices =  (pron_index, pron_index + len(answer1.split()) - 1)
        optionB_indices =  (pron_index, pron_index + len(answer2.split()) - 1)

        text_original = txt1 + " " + pron.strip() + " " + txt2.lstrip()
        text_optionA = txt1 + " " + answer1.strip() + " " + txt2.lstrip()
        text_optionB = txt1 + " " + answer2.strip() + " " + txt2.lstrip()
        print(text_original, "\t", pron, "\t",  pron_index,  "\t", text_optionA, "\t",  optionA_indices, "\t", answer1,
            "\t",  text_optionB, "\t", optionB_indices, "\t", answer2, "\t", correctAnswer, "\t", source)
        answer1 = ''

    elif child.tag.strip() == "schema":
        answer1 = ''
        print("text_original", "\t", "pron", "\t",  "pron_index",  "\t","text_optionA", "\t",  "optionA_indices", "\t", "answerA",
              "\t" "text_optionB", "\t", "optionB_indices", "\t", "answerB", "\t", "correctAnswer","\t", "source")

        skip_once = False
        continue



