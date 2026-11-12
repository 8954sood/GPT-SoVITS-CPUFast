"""Select trainable decoder parameters without modifying inference behavior."""


def configure_trainable(student, scope):
    if scope not in ("all", "last_stage"):
        raise ValueError("Unknown decoder training scope")
    student.requires_grad_(scope == "all")
    if scope == "last_stage":
        for block in student.resblocks[student.stage_offsets[-1]:]:
            block.requires_grad_(True)
        student.conv_post.requires_grad_(True)
    names = [name for name, p in student.named_parameters() if p.requires_grad]
    if not names:
        raise ValueError("Nonempty decoder training scope required")
    return names
