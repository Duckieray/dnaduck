# DNADuck WebbDuck Plugin Package

This folder contains the DNADuck web-app plugin for WebbDuck.

Plugin payload:

- `webapps/dnaduck/plugin.json`
- `webapps/dnaduck/backend.py`
- `webapps/dnaduck/ui/*`

Install using:

```bash
python3 tools/install_webbduck_plugin.py --webbduck-dir /path/to/webbduck --overwrite
```

or:

```bash
python3 tools/install_webbduck_plugin.py --plugins-dir ~/.webbduck/plugins --overwrite
```
