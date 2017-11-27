#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:35:36 2017

@author: eti
"""

import pynini

#money



#measure


back_vowel = pynini.union("u", "o", "a")
neutral_vowel = pynini.union("i", "e")
front_vowel = pynini.union("y", "ö", "ä")
vowel = pynini.union(back_vowel, neutral_vowel, front_vowel)
archiphoneme = pynini.union("A", "I", "E", "O", "U")
consonant = pynini.union("b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n",
                         "p", "q", "r", "s", "t", "v", "w", "x", "z")
sigma_star = pynini.union(vowel, consonant, archiphoneme).closure()

adessive = "llA"
intervener = pynini.union(consonant, neutral_vowel).closure()
adessive_harmony = (pynini.cdrewrite(pynini.transducer("A", "a" ),
                                     back_vowel + intervener, "", sigma_star) *
                    pynini.cdrewrite(pynini.t("A", "ä" ), "", "", sigma_star)
                   ).optimize()


def make_adessive(stem):
  return ((stem + adessive) * adessive_harmony).stringify()



make_adessive("training")




singular_map = pynini.union(
    pynini.transducer("feet", "foot"),
    pynini.transducer("pence", "penny"),
    # Any sequence of bytes ending in "ches" strips the "es";
    # the last argument -1 is a "weight" that gives this analysis
    # a higher priority, if it matches the input.
    sigma_star + pynini.transducer("ches", "ch", -1),
    # Any sequence of bytes ending in "s" strips the "s".
    sigma_star + pynini.transducer("s", ""))


rc = pynini.union(".", ",", "!", ";", "?", " ", "[EOS]")

singularize = pynini.cdrewrite(singular_map, " 1 ",  rc , sigma_star)

def sg(x):
  return pynini.shortestpath(
      pynini.compose(x.strip(), singularize)).stringify()
  
  
#sg("The current temperature in New York is 1 degrees")  
sg("That costs just 1 pence")  