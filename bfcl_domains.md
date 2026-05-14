# BFCL v4 Benchmark — Domain Coverage

There is no explicit `domain` field in any data file.
Domains are inferred from function names and descriptions across all 1,853 unique functions.

## Domain Breakdown

| Domain | Unique Functions | Examples |
|--------|-----------------|---------|
| Entertainment | 319 | Movies, music, concerts, events, video games |
| Math & Science | 258 | Arithmetic, geometry, kinematics, physics, statistics |
| File System & OS | 161 | GorillaFileSystem, cmd_controller, bash-style ops |
| Finance & Banking | 102 | Payments, trading, currency, investment, legal |
| Travel & Transport | 83 | Flights, hotels, buses, trains, ride-sharing |
| Web & Security | 76 | Web search, penetration testing, ACL, OSINT |
| Maps & Location | 71 | Geolocation, addresses, nearby search |
| Cloud & DevOps | 62 | AWS, dashboards, service configs, infrastructure |
| Weather & Environment | 61 | Forecasts, snow reports, temperature |
| Food & Dining | 57 | Restaurants, groceries, dietary logging |
| Messaging & Social | 38 | Twitter, MessageAPI, SMS |
| History & Knowledge | 37 | Country facts, sports rankings, prime ministers |
| Database & Storage | 36 | SQL, Postgres, MemoryAPI |
| Smart Home & IoT | 13 | VehicleControlAPI, LG ThinQ, Bluetooth |
| Healthcare & Services | 6 | Appointments, service providers |
| Other / General Purpose | 473 | Mixed crowd-sourced real-world APIs (live_multiple) |

**Total unique functions: 1,853**

## Data Files and Sample Counts

| File | Samples | Type |
|------|---------|------|
| BFCL_v4_simple_python.json | 400 | Non-live single-turn |
| BFCL_v4_multiple.json | 200 | Non-live single-turn |
| BFCL_v4_parallel.json | 200 | Non-live single-turn |
| BFCL_v4_parallel_multiple.json | 200 | Non-live single-turn |
| BFCL_v4_simple_java.json | 100 | Non-live single-turn |
| BFCL_v4_simple_javascript.json | 50 | Non-live single-turn |
| BFCL_v4_live_simple.json | 258 | Live single-turn |
| BFCL_v4_live_multiple.json | 1053 | Live single-turn |
| BFCL_v4_live_parallel.json | 16 | Live single-turn |
| BFCL_v4_live_parallel_multiple.json | 24 | Live single-turn |
| BFCL_v4_irrelevance.json | 240 | Non-live relevance detection |
| BFCL_v4_live_irrelevance.json | 884 | Live relevance detection |
| BFCL_v4_live_relevance.json | 16 | Live relevance detection |
| BFCL_v4_multi_turn_base.json | 200 | Multi-turn |
| BFCL_v4_multi_turn_long_context.json | 200 | Multi-turn |
| BFCL_v4_multi_turn_miss_func.json | 200 | Multi-turn |
| BFCL_v4_multi_turn_miss_param.json | 200 | Multi-turn |
| BFCL_v4_memory.json | 155 | Multi-turn (memory) |
| BFCL_v4_web_search.json | 100 | Multi-turn (web search) |
| BFCL_v4_format_sensitivity.json | 10 | Format sensitivity |

## Multi-Turn API Classes

Multi-turn tests use virtual API environments drawn from these classes:

| Class | Domain |
|-------|--------|
| GorillaFileSystem | File System & OS |
| VehicleControlAPI | Smart Home & IoT |
| TradingBot | Finance & Banking |
| TravelAPI | Travel & Transport |
| MessageAPI | Messaging & Social |
| TwitterAPI | Messaging & Social |
| TicketAPI | Entertainment |
| MathAPI | Math & Science |
| MemoryAPI | Database & Storage |
| WebSearch | Web & Security |

## Notes

- The **"Other / General Purpose"** category (473 functions) is primarily from
  `BFCL_v4_live_multiple.json` — crowd-sourced real-world function definitions
  submitted by users, covering a long tail of domain-specific APIs
  (code analysis, Java introspection, board games, MBTI, restaurant ordering, etc.).

- Non-live files (`BFCL_v4_simple_python`, `BFCL_v4_multiple`, `BFCL_v4_parallel`)
  use SGD (Schema-Guided Dialogue) style function names like `Hotels_2_SearchHouse`,
  `Events_3_FindEvents`, `Flights_4_SearchOnewayFlight`.

- Live files contain real-world, user-submitted function definitions with no
  standardized naming scheme.
