"""Keyword dictionaries, regex patterns and scoring weights.

This is the *knowledge base* of the offline, rule-based extractor. Everything
domain-specific (which words signal a circuit type, which words signal a
schematic vs. a result plot, how figures are scored) lives here so the heuristics
can be tuned in one place without touching the pipeline code.
"""
from __future__ import annotations

import re

# --------------------------------------------------------------------------- #
# Layout / structure regexes
# --------------------------------------------------------------------------- #

# A real figure caption: "Fig. 4. ...", "Figure 4 ...", "Fig. 4(a) ..."
CAPTION_RE = re.compile(r'^\s*(?:Fig\.?|Figure)\s*0*(\d+)\b', re.IGNORECASE)
# A table caption: "TABLE I", "Table 2"
TABLE_RE = re.compile(r'^\s*Table\s+([IVXLC0-9]+)\b', re.IGNORECASE)
# Section header: roman numeral (or letter) + "." + mostly-uppercase title
SECTION_RE = re.compile(r'^\s*([IVX]{1,5})\.\s+([A-Z][A-Z0-9 \-,/&]+)\s*$')
# DOI anywhere in text
DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+')
# In-body figure reference: "Fig. 4", "Figs. 4 and 5", "Figure 4(a)"
FIG_REF_RE = re.compile(r'\bFig(?:s?\.?|ure)\s*0*(\d+)', re.IGNORECASE)

# Lines/blocks that are page furniture and must be discarded.
NOISE_PATTERNS = [
    re.compile(r'authorized licensed use', re.IGNORECASE),
    re.compile(r'downloaded on .* from ieee xplore', re.IGNORECASE),
    re.compile(r'restrictions apply', re.IGNORECASE),
    re.compile(r'^\s*\d{3,4}\s*$'),                      # bare page numbers like "1854"
    re.compile(r'\d{4}-\d{3,4}/\$?\d', ),               # "0018-9480/$25.00 © 2007 IEEE"
    re.compile(r'digital object identifier', re.IGNORECASE),
]

# --------------------------------------------------------------------------- #
# Section classification
# --------------------------------------------------------------------------- #

SECTION_KIND_KEYWORDS = {
    "intro":      ["introduction", "background"],
    "proposed":   ["proposed", "topology", "architecture", "principle",
                   "technique", "concept", "theory", "analysis"],
    "design":     ["design", "implementation", "circuit", "realization",
                   "building block", "schematic"],
    "results":    ["experiment", "measurement", "measured", "result",
                   "performance", "simulation", "characterization", "discussion"],
    "conclusion": ["conclusion", "summary"],
}

# Sections from which to mine circuit context (in priority order).
CONTEXT_SECTION_KINDS = ["proposed", "design", "intro"]

# --------------------------------------------------------------------------- #
# Circuit-type vocabulary
# --------------------------------------------------------------------------- #
# Each entry: canonical_name -> list of surface patterns (lowercased substrings
# or short regexes). Order matters only for display tie-breaks.
CIRCUIT_TYPES = {
    "voltage-controlled oscillator (VCO)": ["voltage-controlled oscillator", "voltage controlled oscillator", "vco"],
    "oscillator":                          ["oscillator"],
    "low-noise amplifier (LNA)":           ["low-noise amplifier", "low noise amplifier", "lna"],
    "power amplifier (PA)":                ["power amplifier"],
    "amplifier":                           ["amplifier"],
    "mixer":                               ["mixer", "down-converter", "up-converter", "downconverter", "upconverter"],
    "phase-locked loop (PLL)":             ["phase-locked loop", "phase locked loop", "pll"],
    "frequency divider":                   ["frequency divider", "prescaler", "divider"],
    "frequency multiplier":                ["frequency multiplier", "frequency doubler", "multiplier"],
    "analog-to-digital converter (ADC)":   ["analog-to-digital converter", "adc",
                                            "a/d converter", "a-to-d converter"],
    "digital-to-analog converter (DAC)":   ["digital-to-analog converter", "dac",
                                            "d/a converter", "d-to-a converter"],
    "filter":                              ["filter"],
    "rectifier":                           ["rectifier"],
    "charge pump":                         ["charge pump"],
    "bandgap reference":                   ["bandgap reference", "bandgap"],
    "low-dropout regulator (LDO)":         ["low-dropout", "ldo regulator", "ldo"],
    "comparator":                          ["comparator"],
    "phase shifter":                       ["phase shifter"],
    "switch":                              ["rf switch"],
    "balun":                               ["balun"],
    "transceiver":                         ["transceiver"],
    "receiver":                            ["receiver"],
    "transmitter":                         ["transmitter"],

    # --- Data converters / mixed-signal -------------------------------------
    # Note: the ΣΔ/∆Σ glyphs are usually dropped from extracted text, leaving a
    # bare "Modulator", so the plain word has to be a recognised type.
    "delta-sigma modulator":     ["delta-sigma", "sigma-delta", "ΣΔ", "∆Σ",
                                  "modulator"],
    "track-and-hold / sample-and-hold": ["track-and-hold", "track and hold",
                                         "sample-and-hold", "sample and hold"],
    "equalizer":                 ["equalizer", "equaliser"],
    "demodulator":               ["demodulator"],

    # --- Clocking / timing --------------------------------------------------
    "delay-locked loop (DLL)":   ["delay-locked loop", "delay locked loop", "dll"],
    "clock and data recovery (CDR)": ["clock and data recovery", "data recovery",
                                      "cdr", "clock recovery"],
    "frequency synthesizer":     ["frequency synthesizer", "synthesizer",
                                  "synthesiser"],
    "delay line / delay element": ["delay line", "delay element", "delay cell"],
    "phase interpolator":        ["phase interpolator"],
    "phase detector":            ["phase detector", "phase-frequency detector"],
    "flip-flop":                 ["flip-flop", "flipflop", "d-type flip-flop"],

    # --- High-speed serial links -------------------------------------------
    "multiplexer (MUX)":         ["multiplexer", "mux"],
    "demultiplexer (DEMUX)":     ["demultiplexer", "demux"],
    "PRBS generator":            ["prbs generator", "prbs"],
    "decision circuit":          ["decision circuit", "decision-feedback"],
    "transimpedance amplifier (TIA)": ["transimpedance amplifier",
                                       "trans-impedance amplifier", "tia"],
    "pre-amplifier":             ["preamplifier", "pre-amplifier", "preamp"],
    "termination network":       ["termination resistor", "on-die termination",
                                  "termination"],
    "line / output driver":      ["limiting driver", "line driver",
                                  "display driver", "output driver", "driver"],

    # --- Power management ---------------------------------------------------
    "DC-DC converter":           ["dc-dc converter", "dc–dc converter",
                                  "dc/dc converter", "buck converter",
                                  "boost converter", "switching converter"],
    "voltage regulator":         ["linear regulator", "voltage regulator",
                                  "regulator"],
    "power stage":               ["power stage", "output power stage",
                                  "class-d", "class d"],

    # --- References / bias --------------------------------------------------
    "voltage reference":         ["voltage reference"],
    "current reference":         ["current reference"],
    "temperature reference / sensor": ["temperature reference",
                                       "proportional-to-absolute temperature",
                                       "ptat", "thermometer"],
    "bias circuit":              ["bias circuit", "biasing circuit"],

    # --- Memory -------------------------------------------------------------
    "SRAM":                      ["sram"],
    "DRAM / SDRAM":              ["sdram", "dram"],
    "flash memory":              ["flash memory"],
    "phase-change memory":       ["phase-change memory", "phase change memory"],
    "content-addressable memory (CAM)": ["content-addressable memory",
                                         "content addressable memory", "cam cell"],
    "cache":                     ["cache"],
    "register file":             ["register file"],
    "memory":                    ["memory"],

    # --- Digital / processors ----------------------------------------------
    "microprocessor":            ["microprocessor", "processor"],
    "digital signal processor (DSP)": ["digital signal processor", "dsp"],
    "arithmetic logic unit (ALU)": ["arithmetic logic unit", "alu", "adder"],
    "decoder":                   ["viterbi decoder", "viterbi detector",
                                  "map decoder", "decoder"],
    "microcontroller":           ["microcontroller", "risc"],
    "logic style / logic family": ["logic style", "logic family", "logic gate"],
    "latch":                     ["latch"],

    # --- Sensors / imaging --------------------------------------------------
    "image sensor":              ["image sensor", "vision sensor", "camera",
                                  "imager"],
    "pressure sensor":           ["pressure sensor"],
    "biochemical / gas sensor":  ["biochemical microsensor", "gas detection",
                                  "chemical sensor", "microsensor"],
    "fingerprint sensor":        ["fingerprint sensor"],
    "sensor":                    ["sensor"],

    # --- Passives / devices (device-characterisation papers) ---------------
    "on-chip inductor":          ["spiral inductor", "stacked inductor",
                                  "on-chip inductor", "micromachined inductor",
                                  "inductor"],
    "on-chip transformer":       ["transformer"],
    "transconductor / OTA":      ["operational transconductance amplifier",
                                  "transconductance amplifier", "transconductor",
                                  "ota"],

    # --- Protection ---------------------------------------------------------
    "ESD protection circuit":    ["esd protection", "esd"],

    # --- Generic fall-backs (kept last so specific types win) --------------
    "switch":                    ["antenna switch", "single-pole double-throw",
                                  "spdt", "switch"],
    "buffer":                    ["output buffer", "buffer"],
    "regulator / converter":     ["converter"],
    "integrated circuit / system": ["integrated circuit", "mixed-signal ic"],
}

# Categories that name a broad family rather than a specific circuit. They are
# dropped from the detected list whenever a specific category is also present,
# so the fall-back type name prefers "delta-sigma modulator" over "memory".
GENERIC_CIRCUIT_TYPES = {
    "oscillator", "amplifier", "receiver", "transmitter", "filter",
    "memory", "sensor", "switch", "buffer", "latch", "decoder",
    "regulator / converter", "integrated circuit / system",
    "line / output driver", "on-chip transformer", "bias circuit",
}

# Technology / process descriptors (lowercased substrings).
TECHNOLOGY_TERMS = [
    "cmos", "bicmos", "sige", "gaas", "gan", "soi", "fd-soi", "finfet",
    "hbt", "phemt", "hemt", "mmic", "bjt",
]
PROCESS_NODE_RE = re.compile(r'\b\d{1,3}\s?-?\s?(?:nm|µm|um|μm|micrometer|micron)\b', re.IGNORECASE)

# Application / frequency-band descriptors.
APPLICATION_TERMS = [
    "millimeter-wave", "millimeter wave", "mm-wave", "mmwave",
    "microwave", "rf", "radio frequency", "wireless", "5g", "60 ghz",
    "automotive", "radar", "wlan", "satellite", "biomedical", "iot",
    "ultra-wideband", "uwb", "terahertz", "thz", "baseband", "if",
]

# Sub-block / building-block vocabulary. canonical -> surface patterns.
SUB_BLOCKS = {
    "varactor":            ["varactor"],
    "inductor":            ["inductor", "spiral inductor"],
    "capacitor":           ["capacitor", "mim capacitor", "mom capacitor"],
    "resistor":            ["resistor"],
    "transistor/MOSFET":   ["mosfet", "transistor", "pmos", "nmos"],
    "cross-coupled pair":  ["cross-coupled pair", "cross coupled pair", "cross-coupled", "cross coupled"],
    "current source":      ["current source", "tail current"],
    "current mirror":      ["current mirror"],
    "LC tank / resonator": ["lc-tank", "lc tank", "resonator", "tank"],
    "transmission line":   ["transmission line", "microstrip"],
    "coplanar waveguide (CPW)": ["coplanar waveguide", "cpw"],
    "transformer":         ["transformer"],
    "balun":               ["balun"],
    "output buffer":       ["output buffer", "buffer"],
    "mixer":               ["mixer"],
    "bias circuit":        ["bias circuit", "biasing", "bias network"],
    "switch":              ["switch"],
    "differential pair":   ["differential pair", "differential input pair",
                            "input pair", "differential input stage"],
    "op-amp":              ["operational amplifier", "op-amp", "opamp"],
    # Comparator / data-converter building blocks.
    "pre-amplifier":       ["pre-amplifier", "preamplifier", "pre amplifier", "preamp"],
    "latch":               ["regenerative latch", "regeneration latch",
                            "strongarm latch", "strong-arm latch", "strongarm",
                            "double-tail latch", "double tail latch", "latch"],
    "clock / timing":      ["clock signal", "clocked", "clock phase"],
    "sense amplifier":     ["sense amplifier"],
}

# --------------------------------------------------------------------------- #
# Figure-role scoring
# --------------------------------------------------------------------------- #
# Positive caption keywords -> weight (signals "this is the proposed circuit").
CAPTION_POSITIVE = {
    "circuit topology": 3.5,
    "topology": 2.5,
    "schematic": 3.0,
    "circuit schematic": 3.5,
    "circuit diagram": 3.0,
    "block diagram": 2.0,
    "architecture": 2.0,
    "proposed": 3.0,
    "circuit": 1.0,
    "core": 0.5,
}
# Negative caption keywords -> weight (signals NOT the main proposed schematic).
CAPTION_NEGATIVE = {
    "conventional": -3.0,
    "prior": -2.0,
    "reference": -1.0,
    "equivalent": -2.0,
    "half circuit": -2.0,
    "model": -1.5,
    "simplified": -1.0,
    "simulated": -3.0,
    "measured": -3.0,
    "extracted": -2.5,
    "calculated": -2.0,
    "microphotograph": -3.5,
    "photograph": -3.0,
    "micrograph": -3.0,
    "die photo": -3.0,
    "chip photo": -3.0,
    "layout": -2.0,
    "structure of": -1.5,
    "spectrum": -2.5,
    "phase noise": -2.5,
    "tuning range": -2.5,
    "output power": -2.0,
    "q factor": -2.0,
    "capacitance": -2.0,
    "conductance": -1.5,
    "waveform": -2.0,
    "setup": -2.0,
}

# Section-kind bonus applied to a figure based on the section it lives in.
SECTION_KIND_BONUS = {
    "proposed": 2.0,
    "design": 1.5,
    "intro": -0.5,
    "results": -2.0,
    "conclusion": -1.0,
    "other": 0.0,
}

# Figure-role labels derived from caption signals (for human-readable output).
ROLE_RULES = [
    # (role, any-of keywords)
    ("conventional",        ["conventional", "prior", "reference"]),
    ("analysis_model",      ["equivalent", "half circuit", "model", "simplified"]),
    ("result_plot",         ["simulated", "measured", "extracted", "calculated",
                             "spectrum", "phase noise", "tuning range", "output power",
                             "q factor", "capacitance", "conductance", "waveform"]),
    ("layout_photo",        ["microphotograph", "photograph", "micrograph",
                             "die photo", "chip photo", "layout", "structure of"]),
    ("proposed_schematic",  ["circuit topology", "topology", "schematic",
                             "circuit diagram", "block diagram", "architecture"]),
    ("sub_block_schematic", ["circuit", "buffer", "core"]),
]

# Reference-count weight: each in-body reference adds this (capped).
REF_COUNT_WEIGHT = 0.4
REF_COUNT_CAP = 2.0

# Confidence model: how big the top-vs-second-best gap must be for high confidence.
CONF_STRONG_GAP = 3.0
CONF_MIN_TOP_SCORE = 3.0

# Rendering DPI for figure crops.
RENDER_DPI = 200
# Vertical padding (pt) added above a figure region and below its caption.
CROP_PAD_TOP = 6.0
CROP_PAD_BELOW_CAPTION = 2.0
