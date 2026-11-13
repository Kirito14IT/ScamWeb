# Security Policy

## Supported Code

Security fixes are applied to the current `main` branch and the most recent
experiment tag. Historical experiment tags remain immutable; a security fix
that changes an experiment contract receives a new tag.

## Reporting a Vulnerability

Do not open a public issue for vulnerabilities, leaked credentials, unsafe
archive handling, or paths that expose private data. Use GitHub's private
[security advisory form](https://github.com/nlixsys/ScamWeb/security/advisories/new)
and include the affected commit, platform, reproduction steps, and impact.

Never include API keys, raw scam pages, private URLs, or personal information in
a report. Revoke a credential immediately if it may have been exposed.

## Untrusted Dataset Content

All collected PNG, MHTML, HTML-derived text, URLs, QR codes, and model outputs
are untrusted research inputs.

- Keep the raw dataset outside the repository.
- Do not open MHTML in a privileged browser profile.
- Disable automatic downloads, scripts, extensions, and credential managers.
- Do not follow links, scan QR codes, call phone numbers, or contact accounts
  shown in the dataset.
- Run parsing and recapture in an isolated, disposable environment.
- Treat model-generated legal or attribution claims as unverified text.
- Review archives for path traversal, symbolic links, and unexpected members
  before extraction.

The server and Windows delivery scripts enforce checksums and archive
allowlists. Do not bypass these checks for formal experiments.
