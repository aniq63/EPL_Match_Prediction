// Team crest lookup, keyed by normalized club name.
// Covers standard PL names + the common short/ESPN/API name variants so
// Dashboard (football-data.org names), Predictions/Analytics (ESPN/Understat
// names from the ML pipeline), and History (legacy static names) all resolve
// to the same crest.
const CREST_URLS = {
  "arsenal": "https://upload.wikimedia.org/wikipedia/en/5/53/Arsenal_FC.svg",
  "aston villa": "https://upload.wikimedia.org/wikipedia/en/9/9f/Aston_Villa_FC_crest_%282016%29.svg",
  "bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
  "afc bournemouth": "https://upload.wikimedia.org/wikipedia/en/e/e5/AFC_Bournemouth_%282013%29.svg",
  "brentford": "https://upload.wikimedia.org/wikipedia/en/2/2a/Brentford_FC_crest.svg",
  "brighton": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
  "brighton & hove albion": "https://upload.wikimedia.org/wikipedia/en/f/fd/Brighton_%26_Hove_Albion_logo.svg",
  "burnley": "https://upload.wikimedia.org/wikipedia/en/6/6d/Burnley_F.C._Logo.svg",
  "chelsea": "https://upload.wikimedia.org/wikipedia/en/c/cc/Chelsea_FC.svg",
  "crystal palace": "https://upload.wikimedia.org/wikipedia/en/0/0c/Crystal_Palace_FC_logo_%282022%29.svg",
  "everton": "https://upload.wikimedia.org/wikipedia/en/7/7c/Everton_FC_logo.svg",
  "fulham": "https://upload.wikimedia.org/wikipedia/en/e/eb/Fulham_FC_%28shield%29.svg",
  "ipswich town": "https://upload.wikimedia.org/wikipedia/en/4/43/Ipswich_Town.svg",
  "leeds united": "https://upload.wikimedia.org/wikipedia/en/5/54/Leeds_United_F.C._logo.svg",
  "leicester city": "https://upload.wikimedia.org/wikipedia/en/2/2d/Leicester_City_crest.svg",
  "liverpool": "https://upload.wikimedia.org/wikipedia/en/0/0c/Liverpool_FC.svg",
  "manchester city": "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg",
  "manchester united": "https://upload.wikimedia.org/wikipedia/en/7/7a/Manchester_United_FC_crest.svg",
  "newcastle united": "https://upload.wikimedia.org/wikipedia/en/5/56/Newcastle_United_Logo.svg",
  "nottingham forest": "https://upload.wikimedia.org/wikipedia/en/e/e5/Nottingham_Forest_F.C._logo.svg",
  "sunderland": "https://upload.wikimedia.org/wikipedia/en/6/63/Logo_Sunderland.svg",
  "tottenham": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
  "tottenham hotspur": "https://upload.wikimedia.org/wikipedia/en/b/b4/Tottenham_Hotspur.svg",
  "west ham united": "https://upload.wikimedia.org/wikipedia/en/c/c2/West_Ham_United_FC_logo.svg",
  "wolverhampton wanderers": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
  "wolves": "https://upload.wikimedia.org/wikipedia/en/f/fc/Wolverhampton_Wanderers.svg",
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
};

function normalize(name = "") {
  return name.toLowerCase().trim().replace(/\s+/g, " ");
}

export function getCrest(teamName) {
  if (!teamName) return null;
  const key = normalize(teamName);
  if (CREST_URLS[key]) return CREST_URLS[key];

  // Fuzzy fallback: does any known key fully contain the given name or vice versa?
  const match = Object.keys(CREST_URLS).find(
    (k) => k.includes(key) || key.includes(k)
  );
  return match ? CREST_URLS[match] : null;
}

// Deterministic accent color for the initials fallback badge, drawn from the
// FPL palette so unmatched clubs still feel on-brand rather than random.
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
