# Custom Graphiti MCP Server for Arthur

This directory contains a customized version of the [Graphiti MCP Server](https://github.com/getzep/graphiti) with support for custom edge types and edge type mapping.

## What's Different?

The official Graphiti MCP Server Docker image does not pass custom edge types to the Graphiti Core library, resulting in only generic relationships like "RELATES_TO" and "MENTIONS".

Our custom build adds:
- Support for custom edge type definitions
- Edge type mapping between entity pairs
- Domain-specific relationships (WORKS_ON, FIXES, BLOCKS, etc.)

## Modified Files

### Core Changes
- `mcp_server/src/config/schema.py` - Added `EdgeTypeConfig` and `EdgeTypeMapConfig`
- `mcp_server/src/services/queue_service.py` - Pass edge types to Graphiti Core
- `mcp_server/src/graphiti_mcp_server.py` - Initialize and build edge types from config

### Configuration
- Edge types and mappings are defined in `../graphiti-config.yaml` (project root)
- Mounted into the Docker container at `/app/mcp/config/config.yaml`

## Building

From the project root:

```bash
./build-graphiti.sh
```

This builds the Docker image and tags it as `arthur-graphiti:latest`.

## Docker Compose Usage

The custom image is used in `faryal-docker-compose-scale.yml`:

```yaml
graphiti:
  image: arthur-graphiti:latest
  volumes:
    - ./graphiti-config.yaml:/app/mcp/config/config.yaml:ro
```

## Development

To make changes:

1. Edit files in `mcp_server/src/`
2. Rebuild: `./build-graphiti.sh` (from project root)
3. Restart: `docker-compose -f faryal-docker-compose-scale.yml up -d --force-recreate graphiti`
4. Test your changes

## Upstream Sync

This is based on Graphiti MCP Server. To update to a newer version:

1. Check the latest release: https://github.com/getzep/graphiti/releases
2. Update `GRAPHITI_CORE_VERSION` in `../build-graphiti.sh`
3. Rebuild and test
4. May need to re-apply custom changes if upstream changed

## Key Differences from Official Image

| Feature | Official Image | Custom Image |
|---------|---------------|--------------|
| Edge types support | ❌ No | ✅ Yes |
| Edge type mapping | ❌ No | ✅ Yes |
| Custom relationships | ❌ Generic only | ✅ Domain-specific |
| Configuration | Basic | Extended |

## Testing

After building, verify edge types are loaded:

```bash
docker logs arthur-graphiti | grep "edge types"
```

Expected output:
```
Using custom edge types: WORKS_ON, ASSIGNED_TO, REPORTS_TO, ...
Using edge type map with 12 mappings
```

## Documentation

- `../GRAPHITI_CUSTOM_BUILD.md` - Complete documentation
- `../QUICK_START_GRAPHITI.md` - Quick reference
- `../SETUP_GRAPHITI.md` - Setup guide for team members

## License

This is a derivative work of [Graphiti](https://github.com/getzep/graphiti) by Zep AI, licensed under Apache 2.0.

Our modifications are also licensed under Apache 2.0.
