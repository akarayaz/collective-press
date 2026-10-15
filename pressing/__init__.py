"""Pressing and ball recovery on the SkillCorner A-League open dataset.

Entry points:

    from pressing import events, scene, viz

    eng = events.load_engagements(1925299)
    sc = scene.build_scene(1925299, eng.iloc[0])
    viz.save_scene(sc)
"""

__version__ = "0.1.0"
