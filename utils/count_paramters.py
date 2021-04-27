# ported from  from https://github.com/CuriousAI/mean-teacher/blob/master/pytorch/mean_teacher/utils.py
__all__ = ['parameter_count']

def parameter_count(module, verbose=False):
    params = list(module.named_parameters())
    total_count = sum(int(param.numel()) for name, param in params)
    if verbose:
        lines = [
            "",
            "List of model parameters:",
            "=========================",
        ]

        row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
        
        for name, param in params:
            lines.append(row_format.format(
                name=name,
                shape=" * ".join(str(p) for p in param.size()),
                total_size=param.numel()
            ))
        lines.append("=" * 75)
        lines.append(row_format.format(
            name="all parameters",
            shape="sum of above",
            total_size=total_count
        ))
        lines.append("")
        print("\n".join(lines))
    return total_count