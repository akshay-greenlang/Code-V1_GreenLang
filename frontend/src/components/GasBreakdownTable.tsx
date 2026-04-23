/**
 * GasBreakdownTable — reusable per-gas CO2e breakdown table.
 *
 * Takes the v1.2.0 `GasBreakdown` envelope and renders CO2 / CH4 / N2O /
 * HFCs / PFCs / SF6 / NF3 plus biogenic CO2 with their GWP multipliers
 * when provided. Empty entries are hidden. GWP columns are rendered only
 * if any per-gas value is present.
 */
import Paper from "@mui/material/Paper";
import Table from "@mui/material/Table";
import TableBody from "@mui/material/TableBody";
import TableCell from "@mui/material/TableCell";
import TableContainer from "@mui/material/TableContainer";
import TableHead from "@mui/material/TableHead";
import TableRow from "@mui/material/TableRow";
import Typography from "@mui/material/Typography";
import type { GasBreakdown } from "@greenlang/factors-sdk";

export interface GasBreakdownTableProps {
  gas?: GasBreakdown | null;
  /** Unit suffix to show next to every numeric value (e.g. "kg CO2e/kWh"). */
  unit?: string;
}

interface Row {
  label: string;
  key: keyof GasBreakdown;
  gwpKey?: keyof GasBreakdown;
  biogenic?: boolean;
}

const ROWS: Row[] = [
  { label: "CO₂", key: "CO2" },
  { label: "CH₄", key: "CH4", gwpKey: "ch4_gwp" },
  { label: "N₂O", key: "N2O", gwpKey: "n2o_gwp" },
  { label: "HFCs", key: "HFCs" },
  { label: "PFCs", key: "PFCs" },
  { label: "SF₆", key: "SF6" },
  { label: "NF₃", key: "NF3" },
  { label: "Biogenic CO₂", key: "biogenic_CO2", biogenic: true },
];

function fmt(n: number | null | undefined): string {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  if (Math.abs(n) < 0.0001) return n.toExponential(2);
  return n.toLocaleString(undefined, { maximumFractionDigits: 6 });
}

export function GasBreakdownTable({ gas, unit }: GasBreakdownTableProps) {
  if (!gas) {
    return (
      <Typography variant="caption" color="text.secondary">
        No per-gas breakdown available.
      </Typography>
    );
  }

  const visible = ROWS.filter((r) => {
    const v = gas[r.key];
    return typeof v === "number" && !Number.isNaN(v);
  });

  if (visible.length === 0) {
    return (
      <Typography variant="caption" color="text.secondary">
        No per-gas breakdown available.
      </Typography>
    );
  }

  const hasGwp = visible.some((r) => r.gwpKey && typeof gas[r.gwpKey] === "number");

  return (
    <TableContainer component={Paper} variant="outlined">
      <Table size="small" aria-label="Greenhouse gas breakdown">
        <TableHead>
          <TableRow>
            <TableCell>Gas</TableCell>
            <TableCell align="right">Value{unit ? ` (${unit})` : ""}</TableCell>
            {hasGwp && <TableCell align="right">GWP</TableCell>}
            <TableCell>Notes</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {visible.map((r) => {
            const val = gas[r.key] as number;
            const gwp = r.gwpKey ? (gas[r.gwpKey] as number | undefined) : undefined;
            return (
              <TableRow key={r.key as string}>
                <TableCell>{r.label}</TableCell>
                <TableCell align="right">{fmt(val)}</TableCell>
                {hasGwp && (
                  <TableCell align="right">
                    {gwp !== undefined && gwp !== null ? fmt(gwp) : "—"}
                  </TableCell>
                )}
                <TableCell>
                  {r.biogenic && (
                    <Typography variant="caption" color="text.secondary">
                      Biogenic — outside inventory per GHG Protocol Scope guidance.
                    </Typography>
                  )}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </TableContainer>
  );
}

export default GasBreakdownTable;
