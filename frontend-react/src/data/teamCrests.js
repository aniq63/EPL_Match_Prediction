// Team crest lookup, keyed by normalized club name.
// Covers standard PL names + the common short/ESPN/API name variants so
// Dashboard, Predictions, Analytics, and History all resolve correctly.

const CREST_URLS = {
  // Arsenal
  "arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
  "arsenal fc": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",

  // Aston Villa
  "aston villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_FC_crest_%282016%29.svg",
  "aston villa fc": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_FC_crest_%282016%29.svg",
  "villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_FC_crest_%282016%29.svg",

  // Bournemouth
  "bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
  "afc bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",

  // Brentford
  "brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
  "brentford fc": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",

  // Brighton
  "brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
  "brighton & hove albion": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
  "brighton and hove albion": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
  "brighton & hove albion fc": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",

  // Burnley
  "burnley": "https://upload.wikimedia.org/wikipedia/en/6/6d/Burnley_F.C._Logo.svg",
  "burnley fc": "https://upload.wikimedia.org/wikipedia/en/6/6d/Burnley_F.C._Logo.svg",

  // Chelsea
  "chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
  "chelsea fc": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",

  // Crystal Palace
  "crystal palace": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo_%282022%29.svg",
  "crystal palace fc": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo_%282022%29.svg",

  // Everton
  "everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
  "everton fc": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",

  // Fulham
  "fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
  "fulham fc": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",

  // Ipswich Town
  "ipswich town": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",
  "ipswich": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",
  "ipswich town fc": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",

  // Leeds United
  "leeds united": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",
  "leeds": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",
  "leeds utd": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",

  // Leicester City
  "leicester city": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
  "leicester": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
  "leicester city fc": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",

  // Liverpool
  "liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
  "liverpool fc": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",

  // Luton Town
  "luton town": "https://upload.wikimedia.org/wikipedia/en/9/9d/LutonTownFC2009.svg",
  "luton": "https://upload.wikimedia.org/wikipedia/en/9/9d/LutonTownFC2009.svg",

  // Manchester City
  "manchester city": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
  "man city": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
  "mancity": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
  "manchester city fc": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",

  // Manchester United
  "manchester united": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
  "man united": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
  "man utd": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
  "manutd": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
  "manchester united fc": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",

  // Newcastle United
  "newcastle united": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
  "newcastle": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
  "newcastle utd": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
  "newcastle united fc": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",

  // Nottingham Forest
  "nottingham forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
  "nottingham": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
  "nott'm forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
  "nottm forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
  "nottingham forest fc": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",

  // Sheffield United
  "sheffield united": "https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg",
  "sheffield utd": "https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg",
  "sheff utd": "https://upload.wikimedia.org/wikipedia/en/9/9c/Sheffield_United_FC_logo.svg",

  // Southampton
  "southampton": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",
  "southampton fc": "https://upload.wikimedia.org/wikipedia/en/c/c9/FC_Southampton.svg",

  // Sunderland
  "sunderland": "https://upload.wikimedia.org/wikipedia/en/6/63/Logo_Sunderland.svg",
  "sunderland afc": "https://upload.wikimedia.org/wikipedia/en/6/63/Logo_Sunderland.svg",

  // Tottenham Hotspur
  "tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
  "tottenham hotspur": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
  "tottenham hotspur fc": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
  "spurs": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",

  // West Ham United
  "west ham united": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
  "west ham": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
  "west ham utd": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
  "west ham united fc": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",

  // Wolves
  "wolverhampton wanderers": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
  "wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
  "wolverhampton": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
  "wolverhampton wanderers fc": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",

  // Other historical / Championship clubs
  "blackburn rovers": "https://upload.wikimedia.org/wikipedia/en/0/0f/Blackburn_Rovers.svg",
  "coventry city": "https://upload.wikimedia.org/wikipedia/en/6/68/Coventry_City_FC_logo.svg",
  "hull city": "https://upload.wikimedia.org/wikipedia/en/5/54/Hull_City_A.F.C._logo.svg",
  "norwich": "https://upload.wikimedia.org/wikipedia/en/8/8c/Norwich_City.svg",
  "norwich city": "https://upload.wikimedia.org/wikipedia/en/8/8c/Norwich_City.svg",
  "watford": "https://upload.wikimedia.org/wikipedia/en/e/e2/Watford.svg",
  "west bromwich albion": "https://upload.wikimedia.org/wikipedia/en/8/8b/West_Bromwich_Albion.svg",
  "west brom": "https://upload.wikimedia.org/wikipedia/en/8/8b/West_Bromwich_Albion.svg",
  "middlesbrough": "https://upload.wikimedia.org/wikipedia/en/2/2c/Middlesbrough_FC_crest.svg",
  "cardiff": "https://upload.wikimedia.org/wikipedia/en/3/3c/Cardiff_City_crest.svg",
  "cardiff city": "https://upload.wikimedia.org/wikipedia/en/3/3c/Cardiff_City_crest.svg",
  "huddersfield": "https://upload.wikimedia.org/wikipedia/en/5/5a/Huddersfield_Town_A.F.C._logo.png",
  "swansea": "https://upload.wikimedia.org/wikipedia/en/f/f9/Swansea_City_crest.svg",
  "stoke": "https://upload.wikimedia.org/wikipedia/en/2/29/Stoke_City_FC.svg",
  "stoke city": "https://upload.wikimedia.org/wikipedia/en/2/29/Stoke_City_FC.svg",
};

function normalize(name = "") {
  return name.toLowerCase().trim().replace(/['']/g, "").replace(/\s+/g, " ");
}

export function getCrest(teamName) {
  if (!teamName) return null;
  const key = normalize(teamName);
  if (CREST_URLS[key]) return CREST_URLS[key];

  // Try stripping common suffixes like " FC", " AFC", " Football Club"
  const stripped = key.replace(/\b(fc|afc|football club)\b/g, "").trim();
  if (CREST_URLS[stripped]) return CREST_URLS[stripped];

  // Fuzzy fallback match: check if key is substring of known key or vice versa
  const keys = Object.keys(CREST_URLS);
  const match = keys.find(
    (k) => (k.length > 3 && key.includes(k)) || (key.length > 3 && k.includes(key))
  );
  return match ? CREST_URLS[match] : null;
}

const FALLBACK_COLORS = ["#37003C", "#00C96A", "#04B8C2", "#C4006B", "#6A1B74"];

export function getFallbackColor(teamName = "") {
  const key = normalize(teamName);
  let hash = 0;
  for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) >>> 0;
  return FALLBACK_COLORS[hash % FALLBACK_COLORS.length];
}

export function getInitials(teamName = "") {
  const words = teamName.trim().split(/\s+/).filter(Boolean);
  if (words.length === 0) return "?";
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}

