#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 02:38:19 2017

@author: eti
"""

#Telephon , cardinal, ordinal ,fraction, measure, electronic, address, letters

import re
from num2words import num2words
#from nltk.stem.snowball import SnowballStemmer
#import plain_helpers

#stemmer = SnowballStemmer("english")


#########################################################
roman_values = {'M': 1000, 'D': 500, 'C': 100, 'L': 50,
                'X': 10, 'V': 5, 'I': 1}

#digits_set = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#digits_set = set(digits_set)

re_roman = re.compile("[IVXLCDM]+\.?('s)? ?")


############################################
#telephone
def parse_telephone(string):
    after = ""
    for char in string :
        if char.isdigit() : # in #digits_set "
             after = after + " " + parse_cardinal(char)
        elif char in [ ')' ,'-' , ' ' ] :
             after = after + " " + 'sil'
        elif char.isalpha() :
             after = after + " " + char
    return after[1:]

#######################################################
#cardinal

def parse_cardinal(string):
    is_s = False
    if len(string):
        if string[-1] == "s":
            is_s = True
    if re_roman.match(string):
        string = str(_roman_to_int(string))
    after = []
    for char in string:
        if char.isdigit() : # in digits_set:
            after.append(char)
    if len(after):
        after = _cardinal_to_spoken("".join(after))
        if string[0] == "-":
            after = "minus " + after
        if is_s:
            after += "'s"
        return after
    else:
        return string
    
    
def _roman_to_int(string):
    """Convert from Roman numerals to an integer."""
    roman = []
    for char in string:
        if char in roman_values:
            roman.append(char)
    if len(roman):
        numbers = [roman_values[char] for char in roman]
        total = 0
        if len(numbers) > 1:
            for num1, num2 in zip(numbers, numbers[1:]):
                if num1 >= num2:
                    total += num1
                else:
                    total -= num1
            return total + num2
        else:
            return numbers[0]
    else:    
        return string
        
def _cardinal_to_spoken(string):
    middle = num2words(int(string))
    middle = middle.replace(",", "").replace("-", " ")
    middle = middle.split()
    after = []
    for word in middle:
        if word != "and":
            after.append(word)
    return " ".join(after)

###############################################       
#fraction 
    
def parse_fraction(string):
    number = ""
    cardinal = ""
    if "/" in string:
        if "/" in string[1:]:
            numerator, denominator = string.split("/")

            if " " in numerator[:-1]:
                cardinal, numerator = numerator.split()
                cardinal = parse_cardinal(cardinal)
            numerator = parse_cardinal(numerator)
            denominator = parse_ordinals(denominator)

            if denominator == "second":
                if numerator in ["one", "minus one"]:
                    denominator = "half"
                else:
                    denominator = "halve"
            elif denominator == "fourth":
                denominator = "quarter"

            if len(cardinal):
                number += cardinal + " and "
                if numerator == "one":
                    numerator = "a"

            if denominator == "first":
                number += numerator + " over one"
            else:
                number += numerator + " " + denominator
                if numerator not in ["minus one", "one", "a"]:
                    number += "s"
            return number
    return string


def parse_ordinals(string):
    is_roman = False
    is_s = False
    is_aps = False
    if len(string) > 1:
        if string[-2:] == "'s":
            is_aps = True
        elif string[-1] == "s":
            is_s = True
    if re_roman.match(string):
        is_roman = True
        string = str(_roman_to_int(string))
    after = []
    for char in string:
        if char.isdigit() : # in digits_set:
            after.append(char)
    if len(after):
        after = _ordinal_to_spoken("".join(after))
        if is_roman:
            after = " ".join(["the", after])
        elif string[0] == "-":
            after = "minus " + after
        if is_s:
            after += "s"
        elif is_aps:
            after += "'s"
        return after
    else:
        return string
    
def _ordinal_to_spoken(string):
    middle = num2words(int(string), ordinal=True)
    middle = middle.replace(",", "").replace("-", " ")
    middle = middle.split()
    after = []
    for word in middle:
        if word != "and":
            after.append(word)
    return " ".join(after)    
    

################################################    
#measure

measure_number_string = "0123456789-,. "
measure_unit_dict = {"%": "percent", "m": "meters", "km": "kilometers", "mi": "miles", "cm": "centimeters",
                     "mm": "millimeters", "ft": "feet", "kg": "kilograms", "hp": "horsepower", "nm": "nanometers",
                     "ha": "hectares", "\"": "inches", "g": "grams", "GB": "gigabytes", "MB": "megabytes", "cc": "c c",
                     "oz": "ounces", "kW": "kilowatts", "'": "feet", "mph": "miles per hour", "ch": "ch", "Hz": "Hertz",
                     "lbs": "pounds", "lb": "pounds", "yd": "yards", "kt": "knots", "rpm": "revolutions per minute",
                     "h": "hour", "s": "second", "sq mi": "square miles", "mL": "milliliters", "μm": "micrometers",
                     "mA": "milli amperes", "KB": "kilobytes", "MHz": "megahertz", "V": "Volts", "GHz": "gigahertz",
                     "μg": "micrograms"}

#SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")u'\xc2
SUP = {u'\u2070' : '0' , 
       u'\xb9' : '1' ,u'\xb2' : '2' ,
       u'\xb3' : '3' , u'\u2074' : '4' ,
       u'\2075' : '5' , u'\u2076' : '6' ,
       u'\u2077' : '7' , u'\u2078' : '8' ,
       u'\u2079' : '9' } #, '0123456789'}

regex = re.compile("|".join(map(re.escape, SUP.keys())))

def parse_measure(string):
    #string = string.translate(SUP)
    string = regex.sub(lambda mo: SUP[mo.group(0)], string.decode('utf-8'))
    measure = []
    number = []
    for i, char in enumerate(string):
        if char in measure_number_string:
            number.append(char.lower())
        else:
            measure = string[i:]
            break
    number = "".join(number)
    number.replace(" ", "")

    if len(measure) and len(number):
        if "." in number:
            number = parse_decimal(number)
        else:
            number = parse_cardinal(number)

        numerator = None
        denominator = None
        if measure[0] == "/":
            denominator = measure[1:]
        elif "/" in measure:
            numerator, denominator = measure.split("/")
        else:
            numerator = measure

        if numerator is not None:
            if numerator[-1] == "2" and numerator[:-1] in measure_unit_dict:
                numerator = numerator[:-1]
                numerator = " ".join(["square", measure_unit_dict[numerator]])
            elif numerator[-1] == "3" and numerator[:-1] in measure_unit_dict:
                numerator = numerator[:-1]
                numerator = " ".join(["cubic", measure_unit_dict[numerator]])
            elif numerator in measure_unit_dict:
                numerator = measure_unit_dict[numerator]

        if denominator is not None:
            if denominator[:-1] in measure_unit_dict and denominator[-1] == "2":
                denominator = denominator[:-1]
                denominator = " ".join(["square", measure_unit_dict[denominator]])
            elif denominator[-1] == "3" and denominator[:-1] in measure_unit_dict:
                denominator = denominator[:-1]
                denominator = " ".join(["cubic", measure_unit_dict[denominator]])
            elif denominator in measure_unit_dict:
                denominator = measure_unit_dict[denominator]

        if number == "one" and numerator is not None:
            if numerator[-1] == "s":
                numerator = numerator[:-1]

        if denominator is None and numerator is not None:
            return " ".join([number, numerator])
        elif numerator is None and denominator is not None:
            return " ".join([number, "per", denominator])
        elif denominator is not None and numerator is not None:
            return " ".join([number, numerator, "per", denominator])

    elif len(measure):
        if "/" in measure and len(measure.split("/")) == 2:
            numerator, denominator = measure.split("/")
            if not len(numerator):
                if denominator in measure_unit_dict:
                    return " ".join(["per", measure_unit_dict[denominator]])
                else:
                    return " ".join(["per", denominator])

    return string

def parse_decimal(string):
    try:
        number = ""
        if ".0" == string:
            number = "point o"
        elif "." in string:
            cardinal = ""
            if string[0] == ".":
                decimal = string[1:]
            else:
                cardinal, decimal = string.split(".")
                cardinal = parse_cardinal(cardinal)
            decimal = parse_digits(decimal)
            if decimal == "o":
                decimal = "zero"
            if len(cardinal) and len(decimal):
                number = " ".join([cardinal, "point", decimal])
            elif not len(cardinal):
                number = " ".join(["point", decimal])
        else:
            number = parse_cardinal(string)
        if "million" in string:
            number = number + " million"
        if "billion" in string:
            number = number + " billion"
        if "thousand" in string:
            number = number + " thousand"
        if "trillion" in string:
            number = number + " trillion"
        return number
    except:
        return string
    
    
#############################################
#electronic    
electronic_digits_dict = {"0": "o", "1": " ".join("one"), "2": " ".join("two"), "3": " ".join("three"),
                          "4": " ".join("four"), "5": " ".join("five"), "6": " ".join("six"), "7": " ".join("seven"),
                          "8": " ".join("eight"), "9": " ".join("nine")}


def parse_electronic(string):
    if string == "::":
        return string
    if string[0] == "#" and len(string) > 1:
        return "hash tag " + string[1:].lower()
    after = []
    if string[:7] == "http://" and len(string) > 7:
        after.append("h t t p colon slash slash")
        for char in string[7:]:
            after.append(_electronic_parse_char_http(char))
        return " ".join(after)

    if string[:8] == "https://" and len(string) > 8:
        after.append("h t t p s colon slash slash")
        for char in string[7:]:
            after.append(_electronic_parse_char_http(char))
        return " ".join(after)

    for char in string:
        after.append(_electronic_parse_char(char))
    return " ".join(after)


def _electronic_parse_char(char):
    if char.isalpha():
        return char.lower()
    elif char == ".":
        return "dot"
    elif char == "/":
        return "s l a s h"
    elif char == "-":
        return "d a s h"
    elif char in electronic_digits_dict:
        return electronic_digits_dict[char]
    else:
        return char


def _electronic_parse_char_http(char):
    if char.isalpha():
        return char.lower()
    elif char == ".":
        return "dot"
    elif char == "/":
        return "slash"
    elif char == "-":
        return "dash"
    elif char in electronic_digits_dict:
        return electronic_digits_dict[char]
    else:
        return char    


#################################################
# address

def parse_address(string):
    letters = []
    numbers = []
    suffix = ""
    trailing_zeros = len(string) - len(string.rstrip('0'))  
    for char in string:
        if char.isalpha():
            letters.append(char.lower())
        if char.isdigit():
            numbers.append(char)
    if len(letters) and len(numbers):
        
        if trailing_zeros > 1 :   #5400 100 0100
                char = '1' + "".join(numbers[-trailing_zeros:])
                suffix = parse_cardinal(char).split(" ")[1]
                numbers = numbers[:trailing_zeros]
        
        if len(numbers) <=2 :   # 07 00  #10
            if numbers[0] == '0' :
                number = "o"
                for num in numbers :
                    number = number + " " + parse_cardinal(num)
            else :  #23 ,34
                number = "".join(numbers)
                number = parse_cardinal(number)
        
        elif len(numbers) > 3 : # 5432  , 01002 
             number = ""
             for num in numbers :
                    number = number + " " + parse_cardinal(num)
             number = number[1:]       
        
        elif len(numbers) == 3 :    #180 , 010 , 001   
                ind , numb = numbers[0] , numbers[1:]
                if ind == '0' :
                    if numb[0] == '0':
                        number = "o"
                        for num in numb :
                            number = number + " " + parse_cardinal(num)
                    else :
                        numb = "".join(numb)
                        number = 'o' +  " " + parse_cardinal(numb)
                else :
                    numb = "".join(numb)
                    number = parse_cardinal(ind) + " " +  parse_cardinal(numb) 
                    
        if number == 'zero' :
           number = " ".join([ 'o' for i in range(len(numbers)) ])   
        
        
        number = number.replace('zero' ,'o')
        letters = " ".join(letters)
        return " ".join([letters, number , suffix])
    
    
    return string    
###############################################
# letters        
    
def parse_letters(string):
    if string == "'":
        return string
    string = string.replace("'", "")
    after = []
    if len(string) == 1:
        if string == "é":
            return "e acute"
        else:
            return string
    for char in string:
        if char.isalpha():
            if char == "é":
                after.append("e acute")
            else:
                after.append(char.lower())
        elif char == "&":
            after.append("and")
    after = " ".join(after)
    if len(string) > 1:
        if (not string[:-1].islower()) and string[-1] == "s":
            after = after[:-2] + "'" + after[-1]
    return after


################################################

if __name__=='__main__':
    
    
    #telephone
    print parse_telephone('12 26-54 00 mtn')
    print parse_telephone('(2001)00')
    
    #cardinal
    print parse_cardinal('XIX')
    print parse_cardinal('-5s')
    print parse_cardinal('5400')
    
    #ordinal
    print parse_ordinals('21st')
    print parse_ordinals("III's")
    
    #fraction
    print parse_fraction('10/59')
    print parse_fraction("2 1/2")
    
    #measure
    print parse_measure("20.07 km²")

    
    #electronic
    print parse_electronic("2008Achieve.org") 
    print parse_electronic("::")
    print parse_electronic("http://jama.ama-assn.org")
    
    #address
    print parse_address("B01003") # b o one o o three
    print parse_address("I-5400") # I fifty four hundred
    print parse_address("B007") # b o o seven
    print parse_address("B0100") # b o 1 hundred  
    print parse_address("B5432") # b five four three tw0
    print parse_address("B180") # b one eighty
    
    
    #letters
    print parse_letters("M.")
    print parse_letters("af's")
    print parse_letters("us")