// Static Premier League champions history (1992/93 – 2024/25).
// Ported from the legacy frontend's hardcoded winners list.
export const CHAMPIONS = [
  { season: "2024/25", team: "Liverpool", manager: "Arne Slot", pts: 82 },
  { season: "2023/24", team: "Manchester City", manager: "Pep Guardiola", pts: 91 },
  { season: "2022/23", team: "Manchester City", manager: "Pep Guardiola", pts: 89 },
  { season: "2021/22", team: "Manchester City", manager: "Pep Guardiola", pts: 93 },
  { season: "2020/21", team: "Manchester City", manager: "Pep Guardiola", pts: 86 },
  { season: "2019/20", team: "Liverpool", manager: "Jürgen Klopp", pts: 99 },
  { season: "2018/19", team: "Manchester City", manager: "Pep Guardiola", pts: 98 },
  { season: "2017/18", team: "Manchester City", manager: "Pep Guardiola", pts: 100 },
  { season: "2016/17", team: "Chelsea", manager: "Antonio Conte", pts: 93 },
  { season: "2015/16", team: "Leicester City", manager: "Claudio Ranieri", pts: 81 },
  { season: "2014/15", team: "Chelsea", manager: "José Mourinho", pts: 87 },
  { season: "2013/14", team: "Manchester City", manager: "Manuel Pellegrini", pts: 86 },
  { season: "2012/13", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 89 },
  { season: "2011/12", team: "Manchester City", manager: "Roberto Mancini", pts: 89 },
  { season: "2010/11", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 80 },
  { season: "2009/10", team: "Chelsea", manager: "Carlo Ancelotti", pts: 86 },
  { season: "2008/09", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 90 },
  { season: "2007/08", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 87 },
  { season: "2006/07", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 89 },
  { season: "2005/06", team: "Chelsea", manager: "José Mourinho", pts: 91 },
  { season: "2004/05", team: "Chelsea", manager: "José Mourinho", pts: 95 },
  { season: "2003/04", team: "Arsenal", manager: "Arsène Wenger", pts: 90 },
  { season: "2002/03", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 83 },
  { season: "2001/02", team: "Arsenal", manager: "Arsène Wenger", pts: 87 },
  { season: "2000/01", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 80 },
  { season: "1999/00", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 91 },
  { season: "1998/99", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 79 },
  { season: "1997/98", team: "Arsenal", manager: "Arsène Wenger", pts: 78 },
  { season: "1996/97", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 75 },
  { season: "1995/96", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 82 },
  { season: "1994/95", team: "Blackburn Rovers", manager: "Kenny Dalglish", pts: 89 },
  { season: "1993/94", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 92 },
  { season: "1992/93", team: "Manchester United", manager: "Sir Alex Ferguson", pts: 84 },
];

export function getTitleLeaders() {
  const counts = {};
  CHAMPIONS.forEach((c) => {
    counts[c.team] = (counts[c.team] || 0) + 1;
  });
  return Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6);
}
