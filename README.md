# Coach Kal — AI film review for water polo

Coach Kal is an offline water polo film review project. The repository contains frame extraction and detector training scripts, plus a generator for reviewing the existing pilot. The shot and goal timeline is planned in the roadmap.

- [Roadmap](docs/roadmap.md)
- [Implementation guide](docs/implementation.md)

To generate the branded pilot review, run `tools/make_review.py` with the pilot Python environment from a working copy of the local pilot directory. The script reads local metrics and images and writes `review.html`, a contact sheet, and a CSV answer sheet to that directory.
