/* eslint-disable react/prop-types */

const BlogCard = ({ img, title, date, comment }) => {
  return (
    <div className="space-y-4">
        <img
            className="rounded-lg hover:scale-105 transition-transform cursor-pointer h-[80%]"
            src={img}
            alt="post"
        />
        <div className="text-emerald-600 font-medium">
            <span>{date} / </span>
            <span>{comment} Comments</span>
        </div>
        <h3 className="font-bold text-xl">{title}</h3>
    </div>
  )
}

export default BlogCard
