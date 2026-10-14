# Security policy

altc264 parses untrusted video data, so we treat memory-safety and
denial-of-service bugs (crashes, hangs, excessive memory use on crafted
input) as security issues.

## Reporting a vulnerability

Please report it privately, not in a public issue:

- Email **security@altc.tv**, or
- use GitHub's [private vulnerability reporting](../../security/advisories/new).

Include the input that triggers it (or how to make it), the version or
commit, and what you see. We aim to acknowledge reports within three
working days and to agree a disclosure date with you once a fix is ready.

## Supported versions

Until 1.0, fixes go into the latest release only.
