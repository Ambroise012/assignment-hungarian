# Standard library and third-party imports used throughout the pipeline.
# - typing for type hints (readability, static analysis)
# - warnings to emit non-fatal issues while parsing inputs
# - pandas for CSV I/O and tabular manipulations
# - networkx to build and solve a min-cost flow network (assignment model)
# - pathlib for robust filesystem paths
from typing import Dict, Iterable, List, Optional, Tuple
import warnings
import pandas as pd
import networkx as nx
from pathlib import Path


def resolve_data_dir() -> Path:
    """
    Resolve the data directory (../data if inside src/, otherwise ./data).
    Create the folder if it does not exist.
    """
    # Try to infer repository root from the current file path.
    try:
        base_dir = Path(__file__).resolve().parent
        root_dir = base_dir.parent.parent
    except NameError:
        # Fallback when __file__ is undefined (e.g., notebook/REPL).
        root_dir = Path.cwd()
        # If running from src/, go one level up if a data/ sibling exists.
        if root_dir.name == "src" and (root_dir.parent / "data").exists():
            root_dir = root_dir.parent
    # Compute absolute path to data/ and ensure it exists.
    data_dir = (root_dir / "data").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_projects_df(path: Path) -> pd.DataFrame:
    """
    Load projects.csv with columns:
      - id (required, unique)
      - label (optional -> default = id)
      - capacity (optional -> default = 1; must be >= 0)
    """
    # Read CSV as strings, replace NaNs by empty strings for uniform parsing.
    df = pd.read_csv(path, dtype=str).fillna("")
    # Ensure required column presence.
    if "id" not in df.columns:
        raise ValueError("projects.csv must contain column 'id'.")
    # Default label equals id when missing.
    if "label" not in df.columns:
        df["label"] = df["id"]
    # Default capacity equals 1 when missing.
    if "capacity" not in df.columns:
        df["capacity"] = "1"

    # Normalize whitespace on id/label; coerce capacity to integer >= 0.
    df["id"] = df["id"].str.strip()
    df["label"] = df["label"].str.strip()
    df["capacity"] = (
        pd.to_numeric(df["capacity"].replace("", "1"), errors="coerce")
        .fillna(1)
        .astype(int)
    )

    # Validations to guard against malformed inputs.
    if df["id"].duplicated().any():
        dups = df.loc[df["id"].duplicated(), "id"].tolist()
        raise ValueError(f"projects.csv: 'id' must be unique. Duplicates: {dups}")
    if (df["capacity"] < 0).any():
        negs = df.loc[df["capacity"] < 0, "id"].tolist()
        raise ValueError(
            f"projects.csv: 'capacity' must be >= 0. Offending projects: {negs}"
        )

    # Return a clean schema for downstream use.
    return df[["id", "label", "capacity"]]


def _split_semicolon(s: str) -> List[str]:
    # Utility: split a semicolon-separated string, strip whitespace, drop empties.
    return [x.strip() for x in (s or "").split(";") if x.strip()]


def _parse_weighted_prefs(raw: str, valid: set[str]) -> Dict[str, float]:
    """
    Parse weighted preferences in format 'p1:1.5;p2:3' -> {'p1': 1.5, 'p2': 3.0}
    Ignores unknown projects and non-numeric weights.
    """
    # Accumulator for parsed mapping project_id -> weight (as float).
    result: Dict[str, float] = {}
    # Iterate over tokens like "pid:weight".
    for tok in _split_semicolon(raw):
        if ":" not in tok:
            # Warn and skip malformed items.
            warnings.warn(
                f"Malformed weighted preference (missing ':') ignored: '{tok}'",
                RuntimeWarning,
            )
            continue
        # Split into project id and provided weight (string).
        pid, w = tok.split(":", 1)
        pid = pid.strip()
        # Skip tokens referencing unknown project ids.
        if pid not in valid:
            warnings.warn(
                f"Unknown project in weighted prefs ignored: '{pid}'",
                RuntimeWarning,
            )
            continue
        try:
            # Coerce weight to float; store in result map.
            result[pid] = float(w.strip())
        except ValueError:
            # Skip non-numeric weights with a warning.
            warnings.warn(
                f"Non-numeric weight ignored for '{pid}': '{w}'", RuntimeWarning
            )
            continue
    return result


def load_choices_df(path: Path, valid_projects: Iterable[str]) -> pd.DataFrame:
    """
    Load student-choices.csv with columns:
      - student (required)
      - prefs (required), e.g. "p1;p2;p3" or "p1:0;p2:1.5"
      - weight (optional -> default 1; coerced to >= 1)
      - names  (optional); if provided, must contain exactly `weight` names
        separated by ';'. Otherwise auto-generated from 'student'.
    Automatically detects 'ordered' vs 'weighted'.
    Returns columns: key, weight, mode, prefs_ordered, prefs_weighted, names
    """
    # Read raw choices as strings; sanitize NaNs.
    df = pd.read_csv(path, dtype=str).fillna("")
    # Validate required columns.
    required = {"student", "prefs"}
    if not required.issubset(df.columns):
        raise ValueError(
            "student-choices.csv must contain at least 'student' and 'prefs'."
        )

    # Normalize and drop empty student identifiers.
    df["student"] = df["student"].str.strip()
    df = df[df["student"] != ""].copy()

    # Normalize weights: default 1, coerce to int, lower bound at 1.
    if "weight" not in df.columns:
        df["weight"] = "1"
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(1).astype(int)
    df.loc[df["weight"] < 1, "weight"] = 1

    # Build the set of valid project ids for filtering.
    valid = set(valid_projects)

    # Prepare output columns to be filled during parsing loop.
    modes: List[str] = []
    prefs_ordered_col: List[List[str]] = []
    prefs_weighted_col: List[Dict[str, float]] = []
    names_col: List[List[str]] = []

    # Parse each row into either ordered or weighted preference form.
    for _, row in df.iterrows():
        key = row["student"]
        w = int(row["weight"])
        raw_prefs = str(row["prefs"])
        tokens = _split_semicolon(raw_prefs)
        # Heuristic: presence of ':' indicates weighted specification.
        is_weighted = any(":" in tok for tok in tokens)

        if is_weighted:
            # Weighted mode: parse "pid:value" tokens.
            mode = "weighted"
            weights_map = _parse_weighted_prefs(raw_prefs, valid)
            if not weights_map:
                # If parsing yields nothing valid, fallback to ordered via plain ids.
                modes.append("ordered")
                ordered = [p for p in tokens if ":" not in p and p in valid]
                if not ordered:
                    warnings.warn(
                        f"No valid preference found for '{key}'.",
                        RuntimeWarning,
                    )
                prefs_ordered_col.append(ordered)
                prefs_weighted_col.append({})
            else:
                # Store weighted mapping and also a deterministic order by (weight, pid).
                modes.append(mode)
                ordered_pairs = sorted(weights_map.items(), key=lambda kv: (kv[1], kv[0]))
                prefs_ordered_col.append([pid for pid, _ in ordered_pairs])
                prefs_weighted_col.append(weights_map)
        else:
            # Ordered mode: tokens are ranked by position.
            mode = "ordered"
            modes.append(mode)
            ordered = [p for p in tokens if p in valid]
            if not ordered:
                warnings.warn(
                    f"No valid preference found for '{key}'.",
                    RuntimeWarning,
                )
            prefs_ordered_col.append(ordered)
            prefs_weighted_col.append({})

        # Handle optional explicit names replicating the 'weight' multiplicity.
        raw_names = _split_semicolon(row.get("names", ""))
        if raw_names and len(raw_names) != w:
            # If the given names count mismatches the weight, discard them.
            warnings.warn(
                f"'names' provided for '{key}' but length != weight "
                f"({len(raw_names)} != {w}). Ignoring provided names.",
                RuntimeWarning,
            )
            raw_names = []
        if not raw_names:
            # Auto-generate names: either the key itself or key#i for clones.
            raw_names = [f"{key}#{i + 1}" if w > 1 else key for i in range(w)]
        names_col.append(raw_names)

    # Finalize normalized dataframe with standardized column names.
    df = df.rename(columns={"student": "key"})
    df["mode"] = modes
    df["prefs_ordered"] = prefs_ordered_col
    df["prefs_weighted"] = prefs_weighted_col
    df["names"] = names_col
    return df[
        ["key", "weight", "mode", "prefs_ordered", "prefs_weighted", "names"]
    ].reset_index(drop=True)


def _common_graph_skeleton(
    projects_df: pd.DataFrame,
    total_students: int,
    unassigned_label: str,
) -> Tuple[nx.DiGraph, str, str, Dict[str, int], List[str], int]:
    """
    Build the sink side of the flow network (projects -> t), and add source/sink
    nodes with correct global demand.
    """
    # Extract project ids and capacities from the input table.
    projects = projects_df["id"].tolist()
    capacities = dict(zip(projects_df["id"], projects_df["capacity"]))
    cap = {p: int(capacities.get(p, 1)) for p in projects}
    # Compute total capacity to check feasibility against number of students.
    total_cap = sum(cap.values())
    proj_ids = projects[:]
    if total_cap < total_students:
        # If capacity is insufficient, add a virtual "unassigned" project
        # to absorb leftover demand with a capacity gap.
        proj_ids.append(unassigned_label)
        cap[unassigned_label] = total_students - total_cap

    # Keep deterministic iteration order (stability across runs).
    proj_ids = list(proj_ids)

    # Create a directed graph to model min-cost flow assignment.
    g = nx.DiGraph()
    s, t = "_s", "_t"  # Synthetic source/sink names.
    # Flow target equals the number of people to assign (bounded by total capacity).
    flow_target = min(total_students, sum(cap.values()))
    # Encode global supply/demand on source/sink nodes.
    g.add_node(s, demand=-flow_target)
    g.add_node(t, demand=flow_target)
    # Add each project as a node with an edge to sink constrained by its capacity.
    for p in proj_ids:
        g.add_node(p, demand=0)
        g.add_edge(p, t, capacity=cap[p], weight=0)
    return g, s, t, cap, proj_ids, flow_target


def build_graph_unweighted(
    entries_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    rank_cost: Optional[List[float]],
    penalty: Optional[float],
    unassigned_label: str = "__NA__",
) -> Tuple[nx.DiGraph, Dict]:
    """
    Build graph for 'ordered' mode.
    - rank_cost: list of costs per rank (default: 0 for best, 1 for next, ...).
      If the list is too short, extrapolate linearly with slope=1.
    - penalty: cost for any non-listed project (default = max(rank_cost) + 5).
    """
    # Total number of "people" after expanding weights (multiplicity).
    n_people = int(entries_df["weight"].sum())
    # Default cost vector if unspecified: linear penalty by rank position.
    if rank_cost is None:
        max_len = int(entries_df["prefs_ordered"].map(len).max()) if len(entries_df) else 1
        rank_cost = list(range(max(1, max_len)))  # [0, 1, 2, ..., L-1]
    # Default penalty for projects not listed in preferences.
    if penalty is None:
        penalty = (max(rank_cost) if rank_cost else 5) + 5

    # Build the base network (projects -> sink, with capacities).
    g, s, t, cap, proj_ids, flow_target = _common_graph_skeleton(
        projects_df, n_people, unassigned_label
    )

    # Create one entry node per row (group of clones), connect from source with capacity=weight.
    for i, row in entries_df.reset_index(drop=True).iterrows():
        u = f"e{i}"
        g.add_node(u, demand=0)
        g.add_edge(s, u, capacity=int(row["weight"]), weight=0)

    # Connect each entry node to every project with an arc cost reflecting rank.
    for i, row in entries_df.reset_index(drop=True).iterrows():
        u = f"e{i}"
        prefs: List[str] = row["prefs_ordered"]
        # Map project -> rank index for O(1) cost lookup.
        rank_map = {p: r for r, p in enumerate(prefs)}
        for p in proj_ids:
            if p in rank_map:
                r = rank_map[p]
                if r < len(rank_cost):
                    cost = rank_cost[r]
                else:
                    # If beyond provided cost vector, extend linearly.
                    cost = rank_cost[-1] + (r - (len(rank_cost) - 1))
            else:
                # Penalize assignment to non-listed projects.
                cost = penalty
            # Capacity equals the group's weight; weight is the arc cost.
            g.add_edge(u, p, capacity=int(row["weight"]), weight=float(cost))

    # Package metadata useful for decoding the solution later.
    meta = {
        "s": s,
        "t": t,
        "entries": entries_df,
        "cap": cap,
        "flow_target": flow_target,
        "unassigned": unassigned_label,
    }
    return g, meta


def build_graph_weighted(
    entries_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    penalty: Optional[float],
    unassigned_label: str = "__NA__",
    higher_is_better: bool = False,
) -> Tuple[nx.DiGraph, Dict]:
    """
    Build graph for 'weighted' mode.

    Default meaning: 'weight' = cost (lower is better).
    If higher_is_better=True, transform weights into costs using (max - w).

    - penalty: cost for any non-listed project (default 10.0).
    """
    # Total "people" equals the sum of entry weights.
    n_people = int(entries_df["weight"].sum())
    # Default penalty when a project is absent from a student's weighted map.
    if penalty is None:
        penalty = 10.0

    # Build base network with capacities and sink arcs.
    g, s, t, cap, proj_ids, flow_target = _common_graph_skeleton(
        projects_df, n_people, unassigned_label
    )

    # Add entry nodes and connect from source with capacity=weight.
    for i, row in entries_df.reset_index(drop=True).iterrows():
        u = f"e{i}"
        g.add_node(u, demand=0)
        g.add_edge(s, u, capacity=int(row["weight"]), weight=0)

    # For each entry, connect to all projects with costs from the weighted map.
    for i, row in entries_df.reset_index(drop=True).iterrows():
        u = f"e{i}"
        # Copy to dict: mapping project -> provided weight/score.
        wmap: Dict[str, float] = dict(row["prefs_weighted"])
        if higher_is_better and wmap:
            # Convert a "score" (higher better) into a "cost" (lower better).
            mx = max(wmap.values())
            # Non-negative costs: subtract from max.
            wmap = {p: (mx - v) for p, v in wmap.items()}
        for p in proj_ids:
            # Use provided cost if present, otherwise penalize.
            cost = float(wmap[p]) if p in wmap else float(penalty)
            g.add_edge(u, p, capacity=int(row["weight"]), weight=cost)

    # Metadata for solution expansion and reporting.
    meta = {
        "s": s,
        "t": t,
        "entries": entries_df,
        "cap": cap,
        "flow_target": flow_target,
        "unassigned": unassigned_label,
    }
    return g, meta


def solve_min_cost(g: nx.DiGraph) -> Tuple[Dict, float]:
    """
    Solve min-cost flow. Returns (flow_dict, total_cost).
    """
    # Compute a feasible min-cost flow satisfying node demands and capacities.
    flow = nx.min_cost_flow(g, demand="demand", capacity="capacity", weight="weight")
    # Evaluate the total cost of that flow for reporting/comparison.
    cost = nx.cost_of_flow(g, flow, weight="weight")
    return flow, float(cost)


def expand_to_individual_rows(flow: Dict, meta: Dict) -> pd.DataFrame:
    """
    Expand the flow into per-person rows.
    Columns: student, project_id, choice_rank, choice_weight
    """
    # Unpack inputs: raw entries and special label for "unassigned".
    entries_df: pd.DataFrame = meta["entries"]
    unassigned = meta["unassigned"]
    # Accumulate one output row per assigned "clone" (weight unit).
    rows: List[Tuple[str, Optional[str], Optional[int], Optional[float]]] = []

    # Iterate deterministically over original entries to reconstruct assignments.
    for i, row in entries_df.reset_index(drop=True).iterrows():
        u = f"e{i}"
        # Get (project, flow amount) pairs where some units were assigned.
        alloc = [(p, f) for p, f in flow[u].items() if f > 0]
        # Names list contains one name per multiplicity unit.
        names = list(row["names"])
        key = row["key"]
        # Build helpers to retrieve ranks/weights for the assigned project.
        ordered = row.get("prefs_ordered", []) or []
        rank_map = {p: (r + 1) for r, p in enumerate(ordered)}
        wmap: Dict[str, float] = row.get("prefs_weighted", {}) or {}

        # For each assigned unit, emit an individual row.
        for p, k in alloc:
            for _ in range(int(k)):
                # Pop the next available clone name or synthesize a fallback.
                nm = names.pop(0) if names else f"{key}#?"
                # Store None for project if this is the virtual unassigned sink.
                pid = None if p == unassigned else p
                # Rank/weight are only meaningful for real projects.
                choice_rank = rank_map.get(p) if pid is not None else None
                choice_weight = wmap.get(p) if pid is not None else None
                rows.append((nm, pid, choice_rank, choice_weight))

    # Assemble a tidy per-student assignment table.
    return pd.DataFrame(
        rows, columns=["student", "project_id", "choice_rank", "choice_weight"]
    )


def build_student_df(
    assign_df: pd.DataFrame, projects_df: pd.DataFrame, mode: str
) -> pd.DataFrame:
    """
    Enrich assignment with project labels and an 'initial_choice' string.
    """
    # Map project ids to human-readable labels.
    labels = dict(zip(projects_df["id"], projects_df["label"]))
    df = assign_df.copy()
    # Derive the project label, falling back to id or empty string.
    df["project_label"] = df["project_id"].map(labels).fillna(
        df["project_id"].fillna("")
    )

    # Helper: format the "initial_choice" column differently by mode.
    def _fmt(row: pd.Series) -> str:
        r = row.get("choice_rank")
        w = row.get("choice_weight")
        if mode == "weighted":
            # Encode both rank and weight if available as "rank:weight".
            if pd.notna(r) and pd.notna(w):
                return f"{int(r)}:{w:.3f}"
            if pd.notna(r):
                return f"{int(r)}:"
            if pd.notna(w):
                return f":{w:.3f}"
            return ""
        # Unweighted mode: only rank as string (or empty).
        return str(int(r)) if pd.notna(r) else ""

    # Compute and order by student for readability.
    df["initial_choice"] = df.apply(_fmt, axis=1)
    return df.sort_values("student").reset_index(drop=True)


def build_project_df(
    assign_df: pd.DataFrame,
    projects_df: pd.DataFrame,
    include_unassigned: str | bool = "auto",
    unassigned_label: str = "__NA__",
) -> pd.DataFrame:
    """
    Aggregate by project.

    include_unassigned:
      - True  : always add the 'unassigned_label' row
      - False : never add it
      - 'auto': add only if unassigned exist
    """
    # Preserve project order as listed in the input file.
    order = projects_df["id"].tolist()
    labels = dict(zip(projects_df["id"], projects_df["label"]))
    # Group assigned students by project id.
    grouped = assign_df.groupby("project_id")["student"].apply(list).to_dict()
    rows = []
    # Emit rows for all known projects (even if empty).
    for pid in order:
        students = sorted(grouped.get(pid, []))
        rows.append([labels.get(pid, pid), pid, len(students), ";".join(students)])

    # Optionally include an aggregate "unassigned" row.
    add_unassigned = (include_unassigned is True) or (
        include_unassigned == "auto" and unassigned_label in grouped
    )
    if add_unassigned:
        students = sorted(grouped.get(unassigned_label, []))
        rows.append(
            [unassigned_label, unassigned_label, len(students), ";".join(students)]
        )

    # Return a compact per-project summary (label, id, size, roster).
    return pd.DataFrame(
        rows, columns=["project_label", "project_id", "effectif", "students"]
    )


def write_students_csv(df_students: pd.DataFrame, path: Path) -> None:
    # Persist the per-student assignment with key columns in a stable order.
    df = df_students[["student", "project_id", "project_label", "initial_choice"]].copy()
    df.to_csv(path, index=False, encoding="utf-8")


def write_projects_csv(df_projects: pd.DataFrame, path: Path) -> None:
    # Persist the per-project aggregate table to CSV.
    df_projects.to_csv(path, index=False, encoding="utf-8")


def satisfaction_stats(df_students: pd.DataFrame) -> pd.DataFrame:
    """
    Return quick satisfaction metrics.
    """
    # Work on a copy to avoid mutating caller's dataframe.
    s = df_students.copy()
    # Extract numeric rank (the substring before ':'), coercing to NaN when absent.
    s["rank"] = pd.to_numeric(
        s["initial_choice"].str.split(":").str[0],
        errors="coerce",
    )
    # Compute simple KPIs: coverage, unassigned count, median rank, top-1/top-3 rates.
    out = {
        "n": len(s),
        "assigned": int(s["project_id"].notna().sum()),
        "unassigned": int(s["project_id"].isna().sum()),
        "median_rank": float(s["rank"].median()) if len(s) else float("nan"),
        "p_top1": float((s["rank"] == 1).mean()) if len(s) else float("nan"),
        "p_top3": float((s["rank"] <= 3).mean()) if len(s) else float("nan"),
    }
    return pd.DataFrame([out])


def run_pipeline_unweighted(
    data_dir: Path,
    rank_cost: Optional[List[float]] = None,
    penalty: Optional[float] = None,
    unassigned_label: str = "__NA__",
    write_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, Path]]:
    """
    Run the ordered-preferences pipeline end-to-end.
    """
    # Define canonical input/output file locations.
    paths = {
        "projects": data_dir / "projects.csv",
        "choices": data_dir / "student-choices.csv",
        "student_out": data_dir / "assignment_student_unweighted.csv",
        "project_out": data_dir / "assignment_project_unweighted.csv",
    }
    # Load inputs: project catalog and student choices.
    projects_df = load_projects_df(paths["projects"])
    choices_df = load_choices_df(paths["choices"], projects_df["id"].tolist())
    # Build min-cost flow network for ordered preferences.
    g, meta = build_graph_unweighted(
        choices_df, projects_df, rank_cost, penalty, unassigned_label
    )
    # Solve to obtain flow dict and total optimal cost.
    flow, cost = solve_min_cost(g)
    # Expand aggregate flow into per-person assignments.
    assign_df = expand_to_individual_rows(flow, meta)
    # Create student- and project-oriented output tables.
    df_students = build_student_df(assign_df, projects_df, mode="unweighted")
    df_projects = build_project_df(
        assign_df, projects_df, include_unassigned="auto", unassigned_label=unassigned_label
    )
    # Optionally write CSV outputs.
    if write_outputs:
        write_students_csv(df_students, paths["student_out"])
        write_projects_csv(df_projects, paths["project_out"])
    # Return results and paths to artifacts.
    return df_students, df_projects, cost, paths


def run_pipeline_weighted(
    data_dir: Path,
    penalty: Optional[float] = None,
    unassigned_label: str = "__NA__",
    write_outputs: bool = True,
    higher_is_better: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, float, Dict[str, Path]]:
    """
    Run the weighted-preferences pipeline end-to-end.
    """
    # Define canonical input/output file locations.
    paths = {
        "projects": data_dir / "projects.csv",
        "choices": data_dir / "student-choices.csv",
        "student_out": data_dir / "assignment_student_weighted.csv",
        "project_out": data_dir / "assignment_project_weighted.csv",
    }
    # Load inputs: project catalog and student weighted preferences.
    projects_df = load_projects_df(paths["projects"])
    choices_df = load_choices_df(paths["choices"], projects_df["id"].tolist())
    # Build min-cost flow network for weighted preferences (optionally score->cost).
    g, meta = build_graph_weighted(
        choices_df,
        projects_df,
        penalty,
        unassigned_label,
        higher_is_better=higher_is_better,
    )
    # Solve to obtain optimal flow and its objective value.
    flow, cost = solve_min_cost(g)
    # Expand to per-person assignment rows.
    assign_df = expand_to_individual_rows(flow, meta)
    # Create student- and project-oriented output tables.
    df_students = build_student_df(assign_df, projects_df, mode="weighted")
    df_projects = build_project_df(
        assign_df, projects_df, include_unassigned="auto", unassigned_label=unassigned_label
    )
    # Optionally write CSV outputs.
    if write_outputs:
        write_students_csv(df_students, paths["student_out"])
        write_projects_csv(df_projects, paths["project_out"])
    # Return results and paths to artifacts.
    return df_students, df_projects, cost, paths


def run_both(
    data_dir: Optional[Path] = None,
    rank_cost: Optional[List[float]] = None,
    unweighted_penalty: Optional[float] = None,
    weighted_penalty: Optional[float] = None,
    unassigned_label: str = "__NA__",
    write_outputs: bool = True,
    higher_is_better: bool = False,
) -> Dict[str, object]:
    """
    Run both pipelines and return all outputs.
    """
    # Resolve data directory default and run both un/weighted pipelines.
    data_dir = data_dir or resolve_data_dir()
    stu_unw, prj_unw, cost_unw, paths_unw = run_pipeline_unweighted(
        data_dir=data_dir,
        rank_cost=rank_cost,
        penalty=unweighted_penalty,
        unassigned_label=unassigned_label,
        write_outputs=write_outputs,
    )
    stu_w, prj_w, cost_w, paths_w = run_pipeline_weighted(
        data_dir=data_dir,
        penalty=weighted_penalty,
        unassigned_label=unassigned_label,
        write_outputs=write_outputs,
        higher_is_better=higher_is_better,
    )
    # Bundle everything in a single dictionary for easy consumption.
    return {
        "students_unweighted": stu_unw,
        "projects_unweighted": prj_unw,
        "cost_unweighted": cost_unw,
        "paths_unweighted": paths_unw,
        "students_weighted": stu_w,
        "projects_weighted": prj_w,
        "cost_weighted": cost_w,
        "paths_weighted": paths_w,
    }

"""
Export and visualization utilities for assignment graphs.

Includes:
- Graph export to GraphML, GEXF, GPickle (compatible), JSON (node-link),
  and a CSV of edges with positive flow.
- A simple bipartite visualization of the assignment (entries -> projects).

All comments, messages, and output labels are in English.
PEP 8 compliant.
"""
from __future__ import annotations  # Allow postponed evaluation of annotations (forward refs).

# Standard library imports for file formats and filesystem manipulation.
import csv  # Write edge lists (with flow) to CSV.
import json  # Serialize graph to JSON (node-link format).
import pickle  # Fallback serialization for NetworkX objects.
from pathlib import Path  # Path-agnostic filesystem operations.
from typing import Dict, List, Tuple  # Type hints for clarity.

# Third-party imports for plotting and graph handling.
import matplotlib.pyplot as plt  # Matplotlib for visualization.
import networkx as nx  # NetworkX for graph structures and I/O.
from networkx.readwrite import json_graph  # Utilities to convert graphs to JSON schemas.


def _ensure_dir(path: Path) -> None:
    """Ensure a directory exists."""
    # Create directory (and parents) if it does not exist; do nothing if it does.
    path.mkdir(parents=True, exist_ok=True)


def export_graph_models(
    graph: nx.DiGraph,
    meta: Dict,  # kept for API symmetry
    flow: Dict,
    export_dir: Path,
    prefix: str = "model",
) -> Dict[str, Path]:
    """
    Export the given network with flow attributes in multiple formats:
      - GraphML (.graphml)
      - GEXF (.gexf)
      - GPickle (.gpickle) : robust across nx versions (fallback to pickle)
      - JSON node-link (.json)
      - CSV of edges with positive flow (_flow_edges.csv)

    Parameters
    ----------
    graph : nx.DiGraph
        The full flow network.
    meta : Dict
        Metadata dict returned by build_graph_* (unused here).
    flow : Dict
        Flow dictionary as returned by solve_min_cost.
    export_dir : Path
        Destination directory.
    prefix : str
        Filename prefix.

    Returns
    -------
    Dict[str, Path]
        Mapping of format name -> written file path.
    """
    _ensure_dir(export_dir)  # Make sure target directory exists.

    # Work on a copy so we can annotate edges with 'flow' without mutating input.
    graph_copy = graph.copy()
    # For each edge that has flow in the solution, store that as an attribute.
    for u in flow:
        for v, fval in flow[u].items():
            if graph_copy.has_edge(u, v):
                graph_copy[u][v]["flow"] = int(fval)

    # Prepare output paths for all exported artifacts.
    paths = {
        "graphml": export_dir / f"{prefix}.graphml",
        "gexf": export_dir / f"{prefix}.gexf",
        "gpickle": export_dir / f"{prefix}.gpickle",
        "json": export_dir / f"{prefix}.json",
        "csv": export_dir / f"{prefix}_flow_edges.csv",
    }

    # GraphML / GEXF
    # Write the annotated graph to GraphML and GEXF (both widely supported).
    nx.write_graphml(graph_copy, paths["graphml"])
    nx.write_gexf(graph_copy, paths["gexf"])

    # --- GPickle with compatibility across nx versions ---
    try:
        # Prefer NetworkX's own write_gpickle when available (module path may vary).
        from networkx.readwrite.gpickle import (  # type: ignore
            write_gpickle as _write_gpickle,
        )

        _write_gpickle(graph_copy, paths["gpickle"])
    except Exception:
        # If not available or fails, fallback to Python's pickle for robustness.
        with open(paths["gpickle"], "wb") as fh:
            pickle.dump(graph_copy, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # JSON node-link
    # Convert the graph to a node-link structure; specify "links" to avoid future warnings.
    data = json_graph.node_link_data(graph_copy, edges="links")
    # Save prettified JSON with UTF-8 encoding and no ASCII escaping (keep labels readable).
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # CSV of positive-flow edges
    # Create a compact CSV listing only edges that carry positive flow in the solution.
    with paths["csv"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # Header: include flow plus original capacity/weight attributes for context.
        writer.writerow(["u", "v", "flow", "capacity", "weight"])
        # Iterate over flow dict; write one row per edge with fval > 0.
        for u in flow:
            for v, fval in flow[u].items():
                if fval > 0 and graph.has_edge(u, v):
                    writer.writerow(
                        [
                            u,  # Source node id.
                            v,  # Target node id.
                            int(fval),  # Integral flow value on (u, v).
                            graph[u][v].get("capacity", ""),  # Edge capacity if present.
                            graph[u][v].get("weight", ""),  # Edge weight (cost) if present.
                        ]
                    )

    # Return all written paths so callers can report or chain further processing.
    return paths


def visualize_assignment_graph(
    flow: Dict,
    meta: Dict,
    export_path: Path,
    title: str = "Assignment",
    max_labels: int = 60,
) -> Path:
    """
    Create a simple bipartite plot of the assignment graph
    (entries -> projects).

    Parameters
    ----------
    flow : Dict
        Flow dictionary as returned by solve_min_cost.
    meta : Dict
        Metadata dict produced by build_graph_* (must contain 'entries'
        and 'unassigned').
    export_path : Path
        Where to save the PNG figure.
    title : str
        Figure title.
    max_labels : int
        Maximum number of node/edge labels to draw (to avoid clutter).

    Returns
    -------
    Path
        The saved image path.
    """
    # Build a new directed graph "h" containing only entry->project edges with flow.
    h = nx.DiGraph()
    # Access the entries DataFrame to know how many left-side nodes exist.
    entries = meta["entries"]
    _unassigned = meta["unassigned"]  # kept for completeness, not used directly

    # Left nodes = "group" nodes e0, e1, ... (one per entry row).
    left_nodes: List[str] = []
    # Iterate deterministically over entries and create left-side nodes.
    for i, _ in entries.reset_index(drop=True).iterrows():
        u = f"e{i}"  # Node id matches the construction used in the flow network.
        h.add_node(u, bipartite=0)  # Mark as left partition.
        left_nodes.append(u)

    # Keep only edges group -> project (ignore source/sink and other auxiliaries).
    edgelist: List[Tuple[str, str]] = []
    right_nodes_set = set()
    # Loop over the flow solution: retain positive flows from entry nodes to projects.
    for u, outs in flow.items():
        if not str(u).startswith("e"):
            continue  # Skip nodes that are not entry-group nodes.
        for v, fval in outs.items():
            if fval > 0 and v not in ("_s", "_t"):  # Ignore source/sink artifacts.
                h.add_node(v, bipartite=1)  # Ensure project node exists on right side.
                h.add_edge(u, v, weight=int(fval))  # Store flow as 'weight' for drawing.
                edgelist.append((u, v))  # Track edges to style them later.
                right_nodes_set.add(v)  # Collect right-side nodes.

    # Compute bipartite layout positions (manual simple columns).
    pos: Dict[str, Tuple[float, float]] = {}
    # Place left nodes in a vertical column at x=0.
    for idx, u in enumerate(left_nodes):
        pos[u] = (0.0, -idx)

    # Sort right nodes for stable layout and place them at x=1.
    right_nodes = sorted(right_nodes_set)
    for idx, v in enumerate(right_nodes):
        pos[v] = (1.0, -idx)

    # Figure size heuristics: scale by node counts to keep dense graphs readable.
    fig_w = max(8.0, len(right_nodes) * 0.25 + 6.0)
    fig_h = max(6.0, len(left_nodes) * 0.12 + 4.0)

    # Create the figure with computed dimensions.
    plt.figure(figsize=(fig_w, fig_h))
    # Draw left partition nodes as squares.
    nx.draw_networkx_nodes(
        h,
        pos,
        nodelist=left_nodes,
        node_shape="s",
        node_size=200,
        alpha=0.85,
    )
    # Draw right partition nodes as circles.
    nx.draw_networkx_nodes(
        h,
        pos,
        nodelist=right_nodes,
        node_shape="o",
        node_size=300,
        alpha=0.9,
    )

    # Edge widths proportional to assigned flow to highlight stronger matches.
    widths = [1 + 2 * h[u][v]["weight"] for (u, v) in edgelist]
    if edgelist:
        nx.draw_networkx_edges(
            h,
            pos,
            edgelist=edgelist,
            width=widths,
            arrows=False,  # Suppress arrows for a cleaner bipartite look.
            alpha=0.5,
        )

    # Optionally draw labels if the chart would not become cluttered.
    if len(left_nodes) <= max_labels:
        nx.draw_networkx_labels(
            h,
            pos,
            labels={u: u for u in left_nodes},
            font_size=8,
        )
    if len(right_nodes) <= max_labels:
        nx.draw_networkx_labels(
            h,
            pos,
            labels={v: v for v in right_nodes},
            font_size=9,
        )

    # Draw edge labels (flow values) only if there are not too many edges.
    if edgelist and len(edgelist) <= max_labels:
        nx.draw_networkx_edge_labels(
            h,
            pos,
            edge_labels={(u, v): h[u][v]["weight"] for (u, v) in edgelist},
            font_size=7,
        )

    # Finalize plot aesthetics and write to disk.
    plt.title(title)
    plt.axis("off")
    export_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists.
    plt.tight_layout()  # Reduce padding and prevent clipping.
    plt.savefig(export_path, dpi=200)  # Save as PNG with reasonable resolution.
    plt.close()  # Free figure resources (important for batch runs).
    return export_path  # Return the path for caller convenience.

def run_sample_3x3() -> None:
    """Run the 3x3 sample batch end-to-end."""
    # Determine the repository's data directory (creates it if missing).
    base_data_dir = resolve_data_dir()
    # Point to the concrete sample subfolder data/3_sample.
    sample_dir = (base_data_dir / "3_sample").resolve()
    # Ensure the sample directory exists to hold inputs/outputs.
    sample_dir.mkdir(parents=True, exist_ok=True)

    # --- Strict inputs ---
    # Construct absolute paths for the two required CSV inputs.
    src_projects = sample_dir / "3_projects.csv"
    src_choices = sample_dir / "3_student-choices.csv"

    # Guard clause: fail fast if either input file is missing.
    if not src_projects.exists() or not src_choices.exists():
        raise FileNotFoundError(
            "Missing files in "
            f"{sample_dir}: "
            f"{'OK' if src_projects.exists() else 'MISSING: 3_projects.csv'}, "
            f"{'OK' if src_choices.exists() else 'MISSING: 3_student-choices.csv'}"
        )

    # --- Load input data ---
    # Load the project catalog (ids, labels, capacities).
    prj_df = load_projects_df(src_projects)
    # Load student choices and normalize to the expected schema.
    ch_df = load_choices_df(src_choices, prj_df["id"].tolist())

    # === UNWEIGHTED variant (ordered / ranks) ===
    # Build the min-cost flow network for rank-ordered preferences.
    g_unw, meta_unw = build_graph_unweighted(
        entries_df=ch_df,
        projects_df=prj_df,
        rank_cost=None,          # Use default rank costs (0, 1, 2, ...).
        penalty=None,            # Use default penalty for unlisted projects.
        unassigned_label="__NA__",  # Virtual bucket if capacity is insufficient.
    )
    # Solve the min-cost flow to obtain the optimal assignment and its cost.
    flow_unw, cost_unw = solve_min_cost(g_unw)

    # Expand group-level flow into per-student rows.
    assign_unw = expand_to_individual_rows(flow_unw, meta_unw)
    # Build a student-oriented view with labels and initial choice info.
    students_unw_df = build_student_df(assign_unw, prj_df, mode="unweighted")
    # Build a project-oriented aggregation (optionally include unassigned).
    projects_unw_df = build_project_df(
        assign_unw,
        prj_df,
        include_unassigned=True,   # Always show the __NA__ bucket if present.
        unassigned_label="__NA__",
    )

    # Define output file paths for the unweighted results.
    out_student_unw = sample_dir / "assignment_student_unweighted.csv"
    out_project_unw = sample_dir / "assignment_project_unweighted.csv"
    # Persist both tables to CSV.
    write_students_csv(students_unw_df, out_student_unw)
    write_projects_csv(projects_unw_df, out_project_unw)

    # === WEIGHTED variant (explicit weights) ===
    # Build the min-cost flow network for explicitly weighted preferences.
    g_w, meta_w = build_graph_weighted(
        entries_df=ch_df,
        projects_df=prj_df,
        penalty=10.0,              # Default cost when a project isn't rated.
        unassigned_label="__NA__", # Same virtual unassigned bucket.
    )
    # Solve the weighted model to get assignments and total cost.
    flow_w, cost_w = solve_min_cost(g_w)

    # Expand to per-student lines (weighted case).
    assign_w = expand_to_individual_rows(flow_w, meta_w)
    # Build student-oriented table with "rank:weight" string when available.
    students_w_df = build_student_df(assign_w, prj_df, mode="weighted")
    # Build per-project aggregation including __NA__ when present.
    projects_w_df = build_project_df(
        assign_w,
        prj_df,
        include_unassigned=True,
        unassigned_label="__NA__",
    )

    # Define output files for the weighted results.
    out_student_w = sample_dir / "assignment_student_weighted.csv"
    out_project_w = sample_dir / "assignment_project_weighted.csv"
    # Write the weighted results to CSV.
    write_students_csv(students_w_df, out_student_w)
    write_projects_csv(projects_w_df, out_project_w)

    # --- Graph/model exports ---
    # Prepare export directory for graph snapshots and figures.
    export_dir = sample_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export annotated graphs (with 'flow' attributes) and a bipartite PNG for unweighted.
    export_graph_models(
        g_unw, meta_unw, flow_unw, export_dir, prefix="unweighted_model"
    )
    visualize_assignment_graph(
        flow_unw,
        meta_unw,
        export_dir / "unweighted_assignment.png",
        title="Assignment (Unweighted)",
    )

    # Do the same exports for the weighted model.
    export_graph_models(
        g_w, meta_w, flow_w, export_dir, prefix="weighted_model"
    )
    visualize_assignment_graph(
        flow_w,
        meta_w,
        export_dir / "weighted_assignment.png",
        title="Assignment (Weighted)",
    )

    # Pretty console output without leaking absolute directories
    def _pretty_path(path, base):
        """
        Return a relative path when possible, else the filename.
        Helps keep console logs tidy and machine-agnostic.
        """
        try:
            return str(path.relative_to(base))
        except Exception:
            return path.name

    # Summarize completion, objective values, and produced files.
    print("\n[3x3] Done.")
    print(f" - Cost (unweighted): {cost_unw}")
    print(f" - Cost (weighted)  : {cost_w}")
    print(" - Files written:")
    print(f"   {_pretty_path(out_student_unw, sample_dir)}")
    print(f"   {_pretty_path(out_project_unw, sample_dir)}")
    print(f"   {_pretty_path(out_student_w, sample_dir)}")
    print(f"   {_pretty_path(out_project_w, sample_dir)}")
    print(f"   Exports in: {_pretty_path(export_dir, sample_dir)}")