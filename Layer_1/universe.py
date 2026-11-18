"""
GCC TICKER UNIVERSE / Updated May 2026
=======================================
Updated from live stockanalysis.com exchange listings.

New additions vs. previous version are marked with # NEW
Stocks with < 12 months of data are marked # IPO / will auto-skip factor
ranking until minimum history requirement is met (handled by pipeline).

Yahoo Finance suffix reference:
  ADX     → .AD
  DFM     → .DU
  Tadawul → .SR
  QSE     → .QA
  Kuwait  → .KW
"""

TICKERS = {

    # ─────────────────────────────────────────────────────────────────────────
    # ADX / Abu Dhabi Securities Exchange
    # Top 30 by market cap, liquidity-filtered. Suffix: .AD
    # ─────────────────────────────────────────────────────────────────────────
    "ADX": [
        # ── Original universe ────────────────────────────────────────────────
        "IHC.AD",           # International Holding Company      ~850B AED mktcap
        "TAQA.AD",          # Abu Dhabi National Energy Company   ~270B
        "ADNOCGAS.AD",      # ADNOC Gas                           ~251B
        "FAB.AD",           # First Abu Dhabi Bank                ~197B
        "EAND.AD",          # Emirates Telecom (Etisalat)         ~161B
        "ADCB.AD",          # Abu Dhabi Commercial Bank            ~97B
        "ADNOCDRILL.AD",    # ADNOC Drilling                       ~79B
        "ALPHADHABI.AD",    # Alpha Dhabi Holding                  ~79B
        "BOROUGE.AD",       # Borouge                              ~75B
        "ADIB.AD",          # Abu Dhabi Islamic Bank               ~72B
        "ALDAR.AD",         # Aldar Properties                     ~62B
        "ADNOCDIST.AD",     # ADNOC Distribution                   ~46B
        "ADNOCLS.AD",       # ADNOC Logistics & Services           ~38B
        "FERTIGLB.AD",      # Fertiglobe                           ~24B
        "PUREHEALTH.AD",    # Pure Health Holding                  ~22B
        "ADPORTS.AD",       # Abu Dhabi Ports                      ~21B
        "PRESIGHT.AD",      # Presight AI Holding                  ~17B
        "RAKBANK.AD",       # National Bank of Ras Al-Khaimah      ~15B
        "NMDC.AD",          # NMDC Group                           ~15B
        "AMR.AD",           # Americana Restaurants International  ~15B
        "AGILITY.AD",       # Agility Global                       ~13B
        "BURJEEL.AD",       # Burjeel Holdings                     ~6B
        "DANA.AD",          # Dana Gas                             ~6B
        "ADNH.AD",          # Abu Dhabi National Hotels            ~5B
        "ADNIC.AD",         # Abu Dhabi National Insurance         ~4B
        "WAHA.AD",          # Al Waha Capital                      ~3B
        "AGTHIA.AD",        # Agthia Group                         ~3B
        "RAKCEC.AD",        # RAK Ceramics                         ~2B
        "ADSB.AD",          # Abu Dhabi Ship Building              ~2B

        # ── NEW listings (added since last universe update) ──────────────────
        "MODON.AD",         # Modon Holding                        ~50B  # NEW / large cap
        "LULU.AD",          # Lulu Retail Holdings                 ~10B  # NEW / major IPO 2024
        "NMDCENR.AD",       # NMDC Energy                          ~12B  # NEW / ADNOC spinoff
        "2POINTZERO.AD",    # Two Point Zero Group                 ~20B  # NEW / verify YF ticker
        "SPACE42.AD",       # Space42                              ~7B   # NEW / tech/satellite
        "ALEFEDT.AD",       # Alef Education Holding               ~7B   # NEW / edtech IPO
        "EMSTEEL.AD",       # EMSTEEL Building Materials           ~7B   # NEW
        "APEX.AD",          # Apex Investment PSC                  ~12B  # NEW
        "PHX.AD",           # Phoenix Group                        ~5B   # NEW / crypto/fintech
        "ALPHADATA.AD",     # Alpha Data PJSC                      ~1.5B # NEW / tech, IPO 2025
        "ICAP.AD",          # Investcorp Capital                   ~3B   # NEW / alt assets
        "ADNHC.AD",         # ADNH Catering                        ~2B   # NEW
        "ORAN.AD",          # Ooredoo / verify .AD vs .QA suffix
        "ASM.AD",           # Al Seer Marine Supplies              ~2B   # NEW
        "INVICTUS.AD",      # Invictus Investment                  ~2B   # NEW
        "PALMS.AD",         # Palms Sports PJSC                    ~1B   # NEW / IPO 2024
        "SAGA.AD",          # Sagasse Investment Company           ~3B   # NEW
        "NBF.AD",           # National Bank of Fujairah            ~14B  # NEW / was missing
        "MBME.AD",          # MBME GROUP                           ~30B  # NEW / verify YF
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # DFM / Dubai Financial Market
    # Top 25 by market cap, liquidity-filtered. Suffix: .DU
    # ─────────────────────────────────────────────────────────────────────────
    "DFM": [
        # ── Original universe ────────────────────────────────────────────────
        "EMAAR.DU",         # Emaar Properties
        "DEWA.DU",          # Dubai Electricity & Water Authority   IPO 2022
        "DIB.DU",           # Dubai Islamic Bank
        "DU.DU",            # du (Emirates Integrated Telecom)
        "SALIK.DU",         # Salik (Dubai Toll)                    IPO 2022
        "PARKIN.DU",        # Parkin (Dubai Parking)                IPO 2024
        "DFM.DU",           # Dubai Financial Market (exchange itself)
        "DAMAC.DU",         # DAMAC Properties
        "EMAARMALLS.DU",    # Emaar Malls
        "EMAARDEV.DU",      # Emaar Development
        "GFH.DU",           # GFH Financial Group
        "CBD.DU",           # Commercial Bank of Dubai
        "MASQ.DU",          # Mashreqbank
        "TABREED.DU",       # National Central Cooling (Tabreed)
        "GGICO.DU",         # Gulf General Investments
        "SHUAA.DU",         # Shuaa Capital
        "DEYAAR.DU",        # Deyaar Development
        "AMLAK.DU",         # Amlak Finance
        "UNIONPROPERTIES.DU", # Union Properties
        "GFIN.DU",          # Gulf Finance House
        "URBN.DU",          # Urbn
        "EMAARIP.DU",       # Emaar Industries & Investments
        "ALMADINA.DU",      # Al Madina for Finance & Investment

        # ── NEW listings ─────────────────────────────────────────────────────
        "ITHMAR.DU",        # Ithmar Capital                       # NEW / verify YF
        "TECOM.DU",         # TECOM Group                          # NEW / IPO 2022
        "EMPOWER.DU",       # Emirates District Cooling            # NEW / IPO 2023
        "DUCAB.DU",         # Dubai Cable Company                  # NEW / verify YF
        "DUBAIHOLDING.DU",  # Dubai Holding                        # NEW / IPO 2024
        "MULTIBANK.DU",     # Multibank Group                      # NEW / verify if listed
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # TADAWUL / Saudi Stock Exchange
    # MSCI Saudi + top 80 by market cap. Suffix: .SR
    # Uses 4-digit numeric codes as tickers on Yahoo Finance
    # ─────────────────────────────────────────────────────────────────────────
    "TADAWUL": [
        # ── Mega-caps / core ─────────────────────────────────────────────────
        "2222.SR",   # Saudi Aramco
        "2010.SR",   # SABIC
        "1120.SR",   # Al Rajhi Bank
        "1180.SR",   # Saudi National Bank (SNB)
        "7010.SR",   # STC (Saudi Telecom)
        "1050.SR",   # Banque Saudi Fransi
        "1080.SR",   # Arab National Bank
        "1020.SR",   # Bank AlJazira
        "1030.SR",   # Saudi Investment Bank
        "1150.SR",   # Alinma Bank
        "2082.SR",   # ACWA Power
        "2083.SR",   # MARAFIQ
        "2080.SR",   # GASCO
        "5110.SR",   # Saudi Energy (ACWA subsidiary)
        "2160.SR",   # Sipchem
        "2300.SR",   # Saudi Arabia Fertilizers (SAFCO)
        "2320.SR",   # Advanced Petrochemical
        "2290.SR",   # Yanbu National Petrochemicals
        "2350.SR",   # Saudi Kayan
        "2330.SR",   # Advanced Petrochemical
        "2380.SR",   # Petro Rabigh
        "2060.SR",   # Sahara Petrochemical
        "6001.SR",   # Jazan Energy & Development (JIGPC)
        "6010.SR",   # Saudi Electricity Company
        "6020.SR",   # Rabigh Refining & Petrochem
        "6040.SR",   # Saudi Ceramic
        "6060.SR",   # Yanbu Cement
        "6070.SR",   # Al Jouf Cement
        "6090.SR",   # Southern Province Cement
        "7020.SR",   # Etihad Etisalat (Mobily)
        "7030.SR",   # Zain KSA
        "7040.SR",   # Go Telecom
        "8010.SR",   # Tawuniya Insurance
        "8210.SR",   # Bupa Arabia
        "8300.SR",   # Wataniya Insurance
        "4002.SR",   # Saudi Ground Services
        "4007.SR",   # Herfy Food
        "4008.SR",   # Saco
        "4020.SR",   # Almarai
        "4030.SR",   # Saudi Airlines Catering
        "4040.SR",   # Saudi Advanced Industries
        "4050.SR",   # Saudi Fisheries
        "4051.SR",   # Tanmiah Food
        "4061.SR",   # Al Hassan Ghazi Ibrahim Shaker
        "4100.SR",   # Dallah Health
        "4141.SR",   # Dur Hospitality
        "4142.SR",   # Seera Group Holding
        "4161.SR",   # Astra Industrial
        "4163.SR",   # Aldawaa Medical Services
        "4164.SR",   # Al Hammadi
        "4170.SR",   # Middle East Healthcare
        "4180.SR",   # Saudi Pharmaceutical
        "4190.SR",   # Jarir Marketing
        "4200.SR",   # Red Sea International
        "4210.SR",   # Kingdom Holding
        "4240.SR",   # Fawaz Abdulaziz Alhokair
        "4250.SR",   # Saudi Cable
        "4260.SR",   # Al Kathiri Holding
        "4270.SR",   # Maharah HR
        "4280.SR",   # Abdullah Saad Mohammed Abo Moati
        "4290.SR",   # Batic Investments
        "4300.SR",   # Emaar Economic City
        "4310.SR",   # Alhokair
        "4320.SR",   # Al-Bandar Tourism
        "4330.SR",   # Elm Company
        "4340.SR",   # Alkhorayef Water & Power
        "4345.SR",   # Miahona
        "1111.SR",   # Tadawul Group (Saudi Exchange)
        "2081.SR",   # AWPT (Saudi Water Partnership)
        "4081.SR",   # Nayifat Finance
        "4082.SR",   # MRNA (Mranat for Real Estate)
        "4083.SR",   # Tasheel Finance
        "4130.SR",   # Saudi Darb
        "1182.SR",   # Amlak International
        "1183.SR",   # SHL Finance

        # ── NEW listings (Vision 2030 wave, 2023–2025 IPOs) ──────────────────
        "4084.SR",   # Derayah Insurance                          # NEW / IPO 2024
        "4346.SR",   # Lumi Rental Company                        # NEW / IPO 2023
        "4347.SR",   # Al Masane Al Kobra Mining                  # NEW / verify YF
        "2370.SR",   # Middle East Pharmaceutical                 # NEW
        "1214.SR",   # Riyad REIT                                 # NEW / REIT
        "8313.SR",   # Rasan Info Services                        # NEW / insurtech IPO
        "8311.SR",   # Enaya Cooperative Insurance                # NEW
        "4348.SR",   # Americana Restaurants (Tadawul dual-listed) # NEW
        "4349.SR",   # Saned Partners                             # NEW / verify
        "4350.SR",   # Alamar Foods                               # NEW / fast food IPO
        "4351.SR",   # Sela Sport                                 # NEW / events/sports IPO 2023
        "1216.SR",   # NCB Capital REIT                           # NEW / REIT
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # QSE / Qatar Stock Exchange. Suffix: .QA
    # ─────────────────────────────────────────────────────────────────────────
    "QSE": [
        "QNBK.QA",   # Qatar National Bank
        "CBQK.QA",   # Commercial Bank of Qatar
        "QIBK.QA",   # Qatar Islamic Bank
        "ORDS.QA",   # Ooredoo
        "MARK.QA",   # Masraf Al Rayan
        "IHGS.QA",   # Industries Qatar
        "MERS.QA",   # Al Meera Consumer Goods
        "UDCD.QA",   # United Development Company
        "BLDN.QA",   # Baladna
        "IQCD.QA",   # Industries Qatar (check vs IHGS)
        "GISS.QA",   # Gulf International Services
        "QEWS.QA",   # Qatar Electricity & Water
        "QGTS.QA",   # Qatar Gas Transport (Nakilat)
        "NLCS.QA",   # Alijarah Holding
        "QAMC.QA",   # Qatar Aluminium Manufacturing
    ],

    # ─────────────────────────────────────────────────────────────────────────
    # KUWAIT / Boursa Kuwait. Suffix: .KW
    # ─────────────────────────────────────────────────────────────────────────
    "KUWAIT": [
        "NBK.KW",        # National Bank of Kuwait
        "KFH.KW",        # Kuwait Finance House
        "ZAIN.KW",       # Zain (Mobile Telecom)
        "BURGAN.KW",     # Burgan Bank
        "BOUBYAN.KW",    # Boubyan Bank
        "HUMANSOFT.KW",  # Humansoft Holding
        "AGILITY.KW",    # Agility Public Warehousing
        "GBK.KW",        # Gulf Bank
        "WARBA.KW",      # Warba Bank
        "MENA.KW",       # MENA Holding
    ],
}

# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION NOTES / run update_gcc_universe.py to confirm which
# tickers are actually live on Yahoo Finance before committing to pipeline.
# Tickers marked # NEW are candidates; the script validates them.
#
# Known ticker format issues to watch:
#   FERTIGLB.AD    → renamed from FERTIGLOBE.AD (Yahoo Finance uses shortened form)
#   2POINTZERO.AD  → non-standard symbol; verify
#   MBME.AD        → newly listed entity; verify YF coverage
#   ORAN.AD        → Ooredoo is QSE-based (ORDS.QA); .AD suffix may not resolve
#   DFM new names  → several 2024/2025 DFM IPOs have patchy YF coverage
# ─────────────────────────────────────────────────────────────────────────────

# Twelve Data symbol mapping for tickers that yfinance cannot fetch.
# Format: "yf_ticker": ("TD_exchange", "TD_symbol")
TWELVE_DATA_MAP = {
    # ADX tickers (yfinance .AD suffix fails; Twelve Data uses bare symbol on ADX exchange)
    "IHC.AD":           ("ADX", "IHC"),
    "TAQA.AD":          ("ADX", "TAQA"),
    "ADNOCGAS.AD":      ("ADX", "ADNOCGAS"),
    "FAB.AD":           ("ADX", "FAB"),
    "EAND.AD":          ("ADX", "EAND"),
    "ADCB.AD":          ("ADX", "ADCB"),
    "ADNOCDRILL.AD":    ("ADX", "ADNOCDRILL"),
    "ALPHADHABI.AD":    ("ADX", "ALPHADHABI"),
    "BOROUGE.AD":       ("ADX", "BOROUGE"),
    "ADIB.AD":          ("ADX", "ADIB"),
    "ALDAR.AD":         ("ADX", "ALDAR"),
    "ADNOCDIST.AD":     ("ADX", "ADNOCDIST"),
    "ADNOCLS.AD":       ("ADX", "ADNOCLS"),
    "FERTIGLB.AD":      ("ADX", "FERTIGLB"),
    "PUREHEALTH.AD":    ("ADX", "PUREHEALTH"),
    "ADPORTS.AD":       ("ADX", "ADPORTS"),
    "PRESIGHT.AD":      ("ADX", "PRESIGHT"),
    "RAKBANK.AD":       ("ADX", "RAKBANK"),
    "NMDC.AD":          ("ADX", "NMDC"),
    "AMR.AD":           ("ADX", "AMR"),
    "AGILITY.AD":       ("ADX", "AGILITY"),
    "BURJEEL.AD":       ("ADX", "BURJEEL"),
    "DANA.AD":          ("ADX", "DANA"),
    "ADNH.AD":          ("ADX", "ADNH"),
    "ADNIC.AD":         ("ADX", "ADNIC"),
    "WAHA.AD":          ("ADX", "WAHA"),
    "AGTHIA.AD":        ("ADX", "AGTHIA"),
    "RAKCEC.AD":        ("ADX", "RAKCEC"),
    "ADSB.AD":          ("ADX", "ADSB"),
    "MODON.AD":         ("ADX", "MODON"),
    "LULU.AD":          ("ADX", "LULU"),
    "NMDCENR.AD":       ("ADX", "NMDCENR"),
    "SPACE42.AD":       ("ADX", "SPACE42"),
    "ALEFEDT.AD":       ("ADX", "ALEFEDT"),
    "EMSTEEL.AD":       ("ADX", "EMSTEEL"),
    "APEX.AD":          ("ADX", "APEX"),
    "PHX.AD":           ("ADX", "PHX"),
    "ALPHADATA.AD":     ("ADX", "ALPHADATA"),
    "NBF.AD":           ("ADX", "NBF"),
    # DFM tickers (canonical .DU; yfinance alias .AE tried first; TD as final fallback)
    "EMAAR.DU":         ("DFM", "EMAAR"),
    "DEWA.DU":          ("DFM", "DEWA"),
    "DIB.DU":           ("DFM", "DIB"),
    "DU.DU":            ("DFM", "DU"),
    "SALIK.DU":         ("DFM", "SALIK"),
    "PARKIN.DU":        ("DFM", "PARKIN"),
    "DFM.DU":           ("DFM", "DFM"),
    "DAMAC.DU":         ("DFM", "DAMAC"),
    "EMAARMALLS.DU":    ("DFM", "EMAARMALLS"),
    "EMAARDEV.DU":      ("DFM", "EMAARDEV"),
    "GFH.DU":           ("DFM", "GFH"),
    "CBD.DU":           ("DFM", "CBD"),
    "MASQ.DU":          ("DFM", "MASQ"),
    "TABREED.DU":       ("DFM", "TABREED"),
    "GGICO.DU":         ("DFM", "GGICO"),
    "SHUAA.DU":         ("DFM", "SHUAA"),
    "DEYAAR.DU":        ("DFM", "DEYAAR"),
    "AMLAK.DU":         ("DFM", "AMLAK"),
    "TECOM.DU":         ("DFM", "TECOM"),
    "EMPOWER.DU":       ("DFM", "EMPOWER"),
}

# Backwards-compatible aliases for main.py / pipeline.py
# Key names match what those modules expect (old convention).
ALL_EXCHANGES = {
    "ADX":     TICKERS["ADX"],
    "DFM":     TICKERS["DFM"],
    "Tadawul": TICKERS["TADAWUL"],
    "Qatar":   TICKERS["QSE"],
    "Kuwait":  TICKERS["KUWAIT"],
}

EXCHANGE_META = {
    "ADX":     {"country": "UAE",          "msci_status": "Emerging",        "suffix": ".AD"},
    "DFM":     {"country": "UAE",          "msci_status": "Emerging",        "suffix": ".DU"},
    "Tadawul": {"country": "Saudi Arabia", "msci_status": "Emerging",        "suffix": ".SR"},
    "Qatar":   {"country": "Qatar",        "msci_status": "Emerging",        "suffix": ".QA"},
    "Kuwait":  {"country": "Kuwait",       "msci_status": "Emerging (2020)", "suffix": ".KW"},
}
