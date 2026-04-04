import type { Meta, StoryObj } from "@storybook/react";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Stack from "@mui/material/Stack";
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";
import { RunStatusChip } from "./RunStatusChip";
import { ShellErrorAlert } from "./ShellErrorAlert";
import { shellColorTokens, shellFocusRing, shellRadii } from "./tokens";

const meta = {
  title: "Shell UI/Primitives",
  parameters: { layout: "padded" }
} satisfies Meta;

export default meta;

type Story = StoryObj;

export const TokenSwatches: Story = {
  render: () => (
    <Stack direction="row" spacing={2} flexWrap="wrap" useFlexGap>
      {(
        [
          ["primary", shellColorTokens.primary],
          ["secondary", shellColorTokens.secondary],
          ["background.default", shellColorTokens.backgroundDefault],
          ["background.paper", shellColorTokens.backgroundPaper]
        ] as const
      ).map(([label, color]) => (
        <Box key={label}>
          <Box
            sx={{
              width: 120,
              height: 72,
              borderRadius: shellRadii.sm,
              bgcolor: color,
              border: "1px solid rgba(255,255,255,0.12)"
            }}
          />
          <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
            {label}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {color}
          </Typography>
        </Box>
      ))}
    </Stack>
  )
};

export const FocusRingSample: Story = {
  render: () => (
    <Button variant="contained" sx={{ "&:focus-visible": { ...shellFocusRing } }}>
      Focus-visible ring uses shell tokens
    </Button>
  )
};

export const AppChromeSample: Story = {
  render: () => (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1 }}>
          Shell chrome (AppBar overrides)
        </Typography>
        <Button color="inherit" size="small">
          Action
        </Button>
      </Toolbar>
    </AppBar>
  )
};

export const RunStatusChips: Story = {
  render: () => (
    <Stack direction="row" spacing={1} flexWrap="wrap" useFlexGap>
      <RunStatusChip chip="PASS" />
      <RunStatusChip chip="WARN" />
      <RunStatusChip chip="FAIL" />
      <RunStatusChip runState="completed" />
      <RunStatusChip runState="partial_success" />
      <RunStatusChip runState="blocked" />
    </Stack>
  )
};

export const ErrorAlertSample: Story = {
  render: () => (
    <ShellErrorAlert
      envelope={{
        title: "Run encountered errors",
        message: "Primary failure summary",
        details: ["Secondary detail A", "Secondary detail B"]
      }}
    />
  )
};
