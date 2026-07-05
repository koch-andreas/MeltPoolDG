## Changelog Guidelines

All *notable* changes to the project **must** be documented. Every logical change should be recorded in its own dedicated changelog file.

## File Location

Place each changelog entry in the following directory: 

`/doc/changelog/current/`

## File Naming Convention

Each changelog file must follow this naming convention: 

`YEAR_MONTH_DAY_LASTNAME.md`

If the same author submits multiple changelog entries on the same day, append a numeric suffix to distinguish the files:

`2026_05_01_SAM.md`

`2026_05_01_SAM_2.md`


## Changelog Format

The changelog format is based on the principles of [Keep a Changelog](http://keepachangelog.com/).

Entries should clearly describe the purpose and impact of the change and be categorized according to its significance.

**Major Changes**

A change should be classified as MAJOR if it:

- introduces a new notable feature or capability,
- modifies the behavior of an existing feature in a significant way,
- renames parameters,
- ...

**Minor Changes**

A change should be classified as MINOR if it:

- fixes bugs without changing the intended behavior,
- corrects documentation, comments, or error messages,
- improves code quality, readability, or maintainability,
- ...

**Each changelog entry should:**

- clearly explain what changed,
- briefly describe why the change was made (when helpful),
- mention any user-visible impact,
- be concise while providing enough context for users and developers to understand the change,
- avoid implementation details unless they are necessary to understand the impact.

The content of each changelog file should follow one of the categories described below.

## [Master] - YYYY/MM/DD

### Added

- MAJOR/MINOR This PR adds a new feature or a new parameter. [#PR](link)

### Changed

- MAJOR/MINOR This PR changes the behavior of an existing feature of parameter. [#PR](link)

### Deprecated

- MAJOR/MINOR This PR deprecates a feature or a parameter. [#PR](link)

### Removed

- MAJOR/MINOR This PR removes a feature or a parameter. [#PR](link)

### Fixed

- MAJOR/MINOR This PR fixes a bug in an existing feature. [#PR](link)

### Documentation

- MAJOR/MINOR This PR modifies the documentation including the explanation of the parameters, the theory guide or the examples. [#PR](link)
