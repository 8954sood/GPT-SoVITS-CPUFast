import torch

from experiments.student.distill_decoder import generate_with_features
from experiments.student.voice_loss import VoiceLoss
from student_runtime import StudentGenerator


def test_log_voice_identity_silence_and_finite_gradients():
    torch.set_num_threads(2)
    criterion = VoiceLoss()
    target = torch.randn(1, 1, 4096) * 0.03
    mel, stft = criterion(target, target)
    assert mel == 0 and stft == 0

    audio = (target + torch.randn_like(target) * 0.001).requires_grad_()
    mel, stft = criterion(audio, target)
    assert mel > 0 and stft > 0
    (mel + 0.25 * stft).backward()
    assert torch.isfinite(audio.grad).all() and audio.grad.abs().max() > 0
    zero = torch.zeros_like(target)
    mel, stft = criterion(zero, zero)
    assert mel == 0 and stft == 0


def test_training_decoder_forward_matches_exported_forward():
    from module.models import Generator

    teacher = Generator(
        4, "1", [3, 5, 7], [[1, 3, 5]] * 3, [2, 2], 16, [4, 4], gin_channels=3
    ).eval()
    teacher.remove_weight_norm()
    student = StudentGenerator(teacher, [[0, 2], [0, 1, 2]])
    latent, conditioning = torch.randn(1, 4, 16), torch.randn(1, 3, 1)
    with torch.no_grad():
        stock, _ = generate_with_features(teacher, latent, conditioning)
        assert torch.equal(stock, teacher(latent, conditioning))
        actual, _ = generate_with_features(student, latent, conditioning)
        assert torch.equal(actual, student(latent, conditioning))
