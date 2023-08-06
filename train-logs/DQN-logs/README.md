# Test Logs

Log name format: `(batch size)-*(negative reward)-*(model struct)`

`negative reward` default = `20`

`model struct` (default = `origin`)

- `origin`: Input(4) - Dense(32) - Dense(32) - Output(2)

- `origin`: Input(4) - Dense(128) - Dense(64) - Dense(16) - Output(2)

### Completed

| Logs | Episodes |
| ---- | -------- |
| 1    | 30       |
| 2    | 30       |
| 3    | 30       |
| 4    | 30       |
| 6    | 30       |
| 8    | 30       |
| 16   | 30       |
|      |          |

### Waited

| Logs        | Episodes |
| ----------- | -------- |
| 4-20        | 30       |
| 4-50        | 30       |
| 4-10-deeper | 30       |
| 4-20-deeper | 30       |
| 4-50-deeper | 30       |

