import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

export default function DistributionChart({ data, colors }) {
  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height={240}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" cx="50%" cy="50%"
            outerRadius={82} innerRadius={48} paddingAngle={3}>
            {data.map((entry) => <Cell key={entry.name} fill={colors[entry.name]} />)}
          </Pie>
          <Tooltip />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
