from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

try:
    import carla
except ImportError:  # pragma: no cover - only happens outside CARLA runtime
    carla = None


@dataclass
class ActorDims:
    actor_id: int
    type_id: str
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw_deg: float
    dist_to_ego: float
    along_ego: float
    lateral_ego: float


def _require_carla():
    if carla is None:
        raise RuntimeError("carla module not available in this Python environment")


def _bounding_box_dims(actor) -> Dict[str, float]:
    extent = actor.bounding_box.extent
    return {
        'length': float(extent.x * 2.0),
        'width': float(extent.y * 2.0),
        'height': float(extent.z * 2.0),
    }


def _estimate_wheelbase(actor) -> Optional[float]:
    try:
        physics = actor.get_physics_control()
        xs = []
        for wheel in getattr(physics, 'wheels', []):
            pos = getattr(wheel, 'position', None)
            if pos is None:
                continue
            xs.append(float(getattr(pos, 'x', 0.0)))
        if len(xs) >= 2:
            return float(max(xs) - min(xs))
    except Exception:
        pass
    return None


def _actor_signature(actor, ego_tf) -> ActorDims:
    tf = actor.get_transform()
    loc = tf.location
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    right = ego_tf.get_right_vector()

    dx = float(loc.x - ego_loc.x)
    dy = float(loc.y - ego_loc.y)
    along = dx * fwd.x + dy * fwd.y
    lateral = dx * right.x + dy * right.y
    dist = math.hypot(dx, dy)
    dims = _bounding_box_dims(actor)

    return ActorDims(
        actor_id=int(actor.id),
        type_id=str(actor.type_id),
        x=float(loc.x),
        y=float(loc.y),
        z=float(loc.z),
        length=float(dims['length']),
        width=float(dims['width']),
        height=float(dims['height']),
        yaw_deg=float(tf.rotation.yaw),
        dist_to_ego=float(dist),
        along_ego=float(along),
        lateral_ego=float(lateral),
    )


def _find_ego_vehicle(world, ego_id: Optional[int], ego_type_hint: str, ego_type_exact: str):
    all_actors = list(world.get_actors().filter('*'))
    vehicles = list(world.get_actors().filter('vehicle.*'))

    if ego_id is not None:
        for actor in all_actors:
            if int(actor.id) == int(ego_id):
                return actor
        raise RuntimeError(f'ego_id={ego_id} not found among world actors')

    ego_type_hint = (ego_type_hint or '').lower().strip()
    ego_type_exact = (ego_type_exact or '').lower().strip()

    if ego_type_exact:
        exact_all = [a for a in all_actors if str(a.type_id).lower() == ego_type_exact]
        if exact_all:
            exact_all.sort(key=lambda a: a.id)
            return exact_all[0]

    if ego_type_hint:
        hinted_all = [a for a in all_actors if ego_type_hint in str(a.type_id).lower()]
        if hinted_all:
            hinted_all.sort(key=lambda a: a.id)
            return hinted_all[0]

    custom = [a for a in all_actors if 'airtor' in str(a.type_id).lower()]
    if custom:
        custom.sort(key=lambda a: a.id)
        return custom[0]

    if vehicles:
        vehicles.sort(key=lambda a: a.id)
        return vehicles[0]
    return None


def _collect_hint_matches(world, hint: str, limit: int = 20) -> List[Dict[str, Any]]:
    h = (hint or '').lower().strip()
    if not h:
        return []
    rows: List[Dict[str, Any]] = []
    for actor in world.get_actors().filter('*'):
        t = str(actor.type_id).lower()
        if h not in t:
            continue
        tf = actor.get_transform()
        rows.append({
            'actor_id': int(actor.id),
            'type_id': str(actor.type_id),
            'x': float(tf.location.x),
            'y': float(tf.location.y),
            'z': float(tf.location.z),
        })
    rows.sort(key=lambda r: r['actor_id'])
    return rows[: max(1, int(limit))]


def _collect_exact_matches(world, exact_type_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    t = (exact_type_id or '').lower().strip()
    if not t:
        return []
    rows: List[Dict[str, Any]] = []
    for actor in world.get_actors().filter('*'):
        if str(actor.type_id).lower() != t:
            continue
        tf = actor.get_transform()
        rows.append({
            'actor_id': int(actor.id),
            'type_id': str(actor.type_id),
            'x': float(tf.location.x),
            'y': float(tf.location.y),
            'z': float(tf.location.z),
        })
    rows.sort(key=lambda r: r['actor_id'])
    return rows[: max(1, int(limit))]


def _match_trailer_like_actor(actor, trailer_hint: str) -> bool:
    type_id = actor.type_id.lower()
    role_name = ''
    try:
        role_name = str(actor.attributes.get('role_name', '')).lower()
    except Exception:
        role_name = ''

    hint = (trailer_hint or '').lower().strip()
    if hint and (hint in type_id or hint in role_name):
        return True
    if 'airtor' in type_id or 'airtor' in role_name:
        return True
    if 'trailer' in type_id or 'trailer' in role_name:
        return True
    return False


def _find_candidate_trailers(world, ego_actor, max_distance: float, trailer_hint: str):
    ego_tf = ego_actor.get_transform()
    ego_loc = ego_tf.location
    primary = []
    fallback = []
    for actor in world.get_actors().filter('*'):
        if actor.id == ego_actor.id:
            continue
        if not _match_trailer_like_actor(actor, trailer_hint):
            continue
        dist = actor.get_transform().location.distance(ego_loc)
        if dist <= max_distance:
            type_id = actor.type_id.lower()
            role_name = ''
            try:
                role_name = str(actor.attributes.get('role_name', '')).lower()
            except Exception:
                role_name = ''
            if (trailer_hint and (trailer_hint.lower() in type_id or trailer_hint.lower() in role_name)) or 'airtor' in type_id or 'airtor' in role_name:
                primary.append(actor)
            else:
                fallback.append(actor)

    def sort_key(actor):
        tf = actor.get_transform()
        dx = float(tf.location.x - ego_loc.x)
        dy = float(tf.location.y - ego_loc.y)
        along = dx * ego_tf.get_forward_vector().x + dy * ego_tf.get_forward_vector().y
        lateral = abs(dx * ego_tf.get_right_vector().x + dy * ego_tf.get_right_vector().y)
        return (0 if along >= -20.0 else 1, abs(along), lateral, actor.id)

    primary.sort(key=sort_key)
    fallback.sort(key=sort_key)
    return primary + fallback


def inspect_world(
    ego_id: Optional[int],
    ego_type_hint: str,
    ego_type_exact: str,
    max_distance: float,
    trailer_hint: str,
    debug_hint_limit: int,
    spawn_blueprint_id: str,
    spawn_index: int,
) -> Dict[str, Any]:
    _require_carla()

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    temp_actor = None
    spawn_info = ''
    ego_actor = _find_ego_vehicle(world, ego_id=ego_id, ego_type_hint=ego_type_hint, ego_type_exact=ego_type_exact)

    if ego_actor is None:
        spawn_points = list(world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError('no vehicle actors found and map has no spawn points for temporary spawn')

        bp_lib = world.get_blueprint_library()
        blueprint_id = (spawn_blueprint_id or 'vehicle.lincoln.mkz').strip()
        try:
            bp = bp_lib.find(blueprint_id)
        except Exception:
            bp = bp_lib.find('vehicle.lincoln.mkz')
            blueprint_id = 'vehicle.lincoln.mkz'

        idx = int(max(0, min(spawn_index, len(spawn_points) - 1)))
        spawn_tf = spawn_points[idx]
        temp_actor = world.try_spawn_actor(bp, spawn_tf)
        if temp_actor is None:
            raise RuntimeError(f'no vehicle actors found and temporary spawn failed at spawn index {idx} with blueprint {blueprint_id}')
        ego_actor = temp_actor
        spawn_info = f'temporary spawn from {blueprint_id} at spawn_index={idx}'

    exact_matches = _collect_exact_matches(world, ego_type_exact, limit=debug_hint_limit)
    hint_matches = _collect_hint_matches(world, ego_type_hint, limit=debug_hint_limit)

    try:
        ego_tf = ego_actor.get_transform()
        ego_dims = _bounding_box_dims(ego_actor)
        ego_wheelbase = _estimate_wheelbase(ego_actor)

        trailer_candidates = _find_candidate_trailers(world, ego_actor, max_distance=max_distance, trailer_hint=trailer_hint)

        rows: List[ActorDims] = []
        prev_actor = ego_actor
        prev_dims = _bounding_box_dims(prev_actor)
        hitch_gaps: List[float] = []

        for actor in trailer_candidates:
            sig = _actor_signature(actor, ego_tf)
            rows.append(sig)

            center_dist = actor.get_transform().location.distance(prev_actor.get_transform().location)
            gap = float(center_dist - (prev_dims['length'] * 0.5 + sig.length * 0.5))
            hitch_gaps.append(gap)
            prev_actor = actor
            prev_dims = _bounding_box_dims(prev_actor)

        result: Dict[str, Any] = {
            'ego': {
                'actor_id': int(ego_actor.id),
                'type_id': str(ego_actor.type_id),
                'location': {
                    'x': float(ego_tf.location.x),
                    'y': float(ego_tf.location.y),
                    'z': float(ego_tf.location.z),
                },
                'rotation_yaw_deg': float(ego_tf.rotation.yaw),
                'bounding_box': ego_dims,
                'wheelbase_estimate_m': None if ego_wheelbase is None else float(ego_wheelbase),
                'spawn_info': spawn_info,
            },
            'ego_type_exact': ego_type_exact,
            'ego_type_exact_matches': exact_matches,
            'ego_type_hint': ego_type_hint,
            'ego_type_hint_matches': hint_matches,
            'trailer_hint': trailer_hint,
            'trailers_or_nearby_vehicles': [asdict(r) for r in rows],
            'estimated_hitch_gaps_m': [float(g) for g in hitch_gaps],
        }

        return result
    finally:
        if temp_actor is not None and temp_actor.is_alive:
            temp_actor.destroy()


def main():
    parser = argparse.ArgumentParser(description='Read geometry parameters directly from CARLA actors')
    parser.add_argument('--ego-id', type=int, default=None, help='ego actor id')
    parser.add_argument('--ego-type-exact', type=str, default='vehicle.airtor666.airtor666', help='exact ego type_id match')
    parser.add_argument('--ego-type-hint', type=str, default='airtor666', help='ego type substring hint')
    parser.add_argument('--trailer-hint', type=str, default='airtor', help='custom trailer type substring, default airtor')
    parser.add_argument('--debug-hint-limit', type=int, default=20, help='max debug rows for ego-type-hint matches')
    parser.add_argument('--max-distance', type=float, default=80.0, help='max distance for nearby trailers/vehicles')
    parser.add_argument('--spawn-blueprint', type=str, default='vehicle.lincoln.mkz', help='temporary ego blueprint when world has no vehicles')
    parser.add_argument('--spawn-index', type=int, default=0, help='spawn point index for temporary ego vehicle')
    parser.add_argument('--json-out', type=str, default='', help='optional json output path')
    args = parser.parse_args()

    data = inspect_world(
        args.ego_id,
        args.ego_type_hint,
        args.ego_type_exact,
        args.max_distance,
        args.trailer_hint,
        args.debug_hint_limit,
        args.spawn_blueprint,
        args.spawn_index,
    )

    ego = data['ego']
    print('=== Ego Vehicle ===')
    print(f"actor_id: {ego['actor_id']}")
    print(f"type_id: {ego['type_id']}")
    if ego.get('spawn_info'):
        print(f"spawn_info: {ego['spawn_info']}")
    print(f"location: ({ego['location']['x']:.3f}, {ego['location']['y']:.3f}, {ego['location']['z']:.3f})")
    print(f"yaw_deg: {ego['rotation_yaw_deg']:.3f}")
    print(
        f"bounding_box: length={ego['bounding_box']['length']:.3f}m, "
        f"width={ego['bounding_box']['width']:.3f}m, height={ego['bounding_box']['height']:.3f}m"
    )
    if ego['wheelbase_estimate_m'] is not None:
        print(f"wheelbase_estimate: {ego['wheelbase_estimate_m']:.3f}m")
    else:
        print('wheelbase_estimate: unavailable (fallback to bbox-based estimate if needed)')

    print(f"ego_type_exact: {data.get('ego_type_exact', '')}")
    exact_matches = data.get('ego_type_exact_matches', [])
    print(f"ego_type_exact_matches: {len(exact_matches)}")
    for i, row in enumerate(exact_matches):
        print(
            f"  ({i + 1}) id={row['actor_id']} type={row['type_id']} "
            f"loc=({row['x']:.3f}, {row['y']:.3f}, {row['z']:.3f})"
        )

    print(f"ego_type_hint: {data.get('ego_type_hint', '')}")
    hint_matches = data.get('ego_type_hint_matches', [])
    print(f"ego_type_hint_matches: {len(hint_matches)}")
    for i, row in enumerate(hint_matches):
        print(
            f"  ({i + 1}) id={row['actor_id']} type={row['type_id']} "
            f"loc=({row['x']:.3f}, {row['y']:.3f}, {row['z']:.3f})"
        )

    print(f"trailer_hint: {data.get('trailer_hint', '')}")

    print('\n=== Nearby Trailer / Vehicle Chain ===')
    gaps = data['estimated_hitch_gaps_m']
    for i, row in enumerate(data['trailers_or_nearby_vehicles']):
        gap_text = ''
        if i < len(gaps):
            gap_text = f", estimated_hitch_gap={gaps[i]:.3f}m"
        print(
            f"[{i + 1}] id={row['actor_id']} type={row['type_id']} "
            f"dist={row['dist_to_ego']:.3f}m along={row['along_ego']:.3f}m lateral={row['lateral_ego']:.3f}m "
            f"size=({row['length']:.3f} x {row['width']:.3f} x {row['height']:.3f})m{gap_text}"
        )

    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\nJSON saved to: {args.json_out}")


if __name__ == '__main__':
    main()