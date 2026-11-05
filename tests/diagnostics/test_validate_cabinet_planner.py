"""Check that validation does not hide forbidden physical contacts."""
from types import SimpleNamespace

import numpy as np
import pytest
from exploration.diagnostics.validate_cabinet_planner import ContactAudit, contact_category, point_is_active


def test_only_explicit_finger_target_contacts_are_allowed():
    target=("object","cup")
    assert contact_category(("robot","panda_leftfinger"),target)=="forbidden_robot_environment"
    assert contact_category(("robot","panda_leftfinger"),target,allowed_items=("cup",))=="allowed_finger_object"
    assert contact_category(target,("robot","panda_hand"),payload="cup")=="forbidden_robot_environment"
    assert contact_category(("robot","panda_rightfinger"),target,payload="cup")=="allowed_finger_object"


def test_payload_contacts_with_other_objects_and_wall_are_not_ignored():
    assert contact_category(("object","cup"),("object","bowl"),payload="cup")=="forbidden_payload_environment"
    assert contact_category(("wall","wall"),("object","cup"),payload="cup")=="forbidden_payload_environment"
    assert contact_category(("object","bowl"),("cabinet","frame"),payload="cup") is None


def test_initial_support_exception_requires_phase_height_and_horizontal_normal():
    pair=(("object","cup"),("cabinet","frame"))
    kw=dict(payload="cup",position=[.1,0,.672],normal=[0,0,1])
    assert contact_category(*pair,**kw)=="forbidden_payload_environment"
    assert contact_category(*pair,allow_support=True,**kw)=="allowed_initial_support"
    kw["normal"]=[1,0,0]
    assert contact_category(*pair,allow_support=True,**kw)=="forbidden_payload_environment"
    kw.update(normal=[0,0,1],position=[.1,0,.68])
    assert contact_category(*pair,allow_support=True,**kw)=="forbidden_payload_environment"


def test_robot_self_and_nonbase_ground_contact_are_not_filtered():
    assert contact_category(("robot","panda_link5"),("robot","panda_link6"))=="forbidden_robot_self"
    assert contact_category(("robot","panda_link0"),("ground","ground"))=="allowed_fixed_base_ground"
    assert contact_category(("robot","panda_hand"),("ground","ground"))=="forbidden_robot_environment"


@pytest.mark.parametrize("separation,impulse,active",[
    (.02,[0,0,0],False),(.001,[.01,0,0],False),
    (0,[.001,0,0],True),(-1e-7,[0,0,0],False),(-2e-5,[0,0,0],True),
])
def test_speculative_manifolds_are_distinct_from_touch_and_penetration(separation,impulse,active):
    assert bool(point_is_active(separation,np.asarray(impulse))) is active


@pytest.mark.parametrize("replace_hook", [False, True])
def test_contact_audit_close_restores_hook_without_writing_closed_stream(tmp_path, replace_hook):
    calls=[]
    original=lambda:calls.append('original')
    actor=lambda name:SimpleNamespace(name=name,_bodies=[SimpleNamespace(name=name)])
    raw=SimpleNamespace(
        agent=SimpleNamespace(robot=SimpleNamespace(get_links=lambda:[])),
        cabinet=SimpleNamespace(get_links=lambda:[]),cabinet_items=[],
        ground=actor('ground'),wall=actor('wall'),
        CABINET_POSE=SimpleNamespace(p=np.zeros(3)),sim_timestep=.01,
        scene=SimpleNamespace(get_contacts=lambda:[]),_after_simulation_step=original)
    path=tmp_path/'contacts.jsonl'
    audit=ContactAudit(raw,path,max_substeps=3)
    raw._after_simulation_step()
    assert calls==['original'] and audit.substeps==1
    replacement=lambda:calls.append('replacement')
    if replace_hook:raw._after_simulation_step=replacement
    audit.close()
    assert audit.stream.closed
    assert raw._after_simulation_step is (replacement if replace_hook else original)
    raw._after_simulation_step()
    assert calls==['original','replacement' if replace_hook else 'original']
    assert audit.substeps==1
    assert len(path.read_text().splitlines())==1
